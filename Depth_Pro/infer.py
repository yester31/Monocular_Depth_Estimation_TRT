# by yhpark 2025-7-16
# check Depth Pro pytorch model inference performance
import os
import torch
import cv2 
import numpy as np
import time 
from matplotlib import pyplot as plt

import depth_pro

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

def show_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[MDET] allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

show_memory_usage()
torch.cuda.empty_cache()
show_memory_usage()

def set_model():
    # Model Config
    CUSTOM_MONODEPTH_CONFIG_DICT = depth_pro.depth_pro.DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=f"{CUR_DIR}/ml-depth-pro/checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )

    # Load model and preprocessing transform
    model, transform = depth_pro.depth_pro.create_model_and_transforms(config=CUSTOM_MONODEPTH_CONFIG_DICT, device=DEVICE, precision=torch.half)
    model.eval()
    return model, transform

def infer_performace(model):
    print('[MDET] Start inference performace check...')

    dummy_input = torch.randn(1, 3, model.img_size, model.img_size).to(DEVICE).half()

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model.infer(dummy_input)
    torch.cuda.synchronize()

    # check FPS 
    iteration = 20
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            _ = model.infer(dummy_input)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f'[MDET] {iteration} iterations time ({dummy_input.shape}): {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

def main():
    print('[MDET] Load model & image')
    model, transform = set_model()

    input_size = model.img_size # 1536 
    interpolation_mode = "bilinear"
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    image_rgb, _, f_px0 = depth_pro.utils.load_rgb(image_path) # RGB
    # ===================================================================
    print('[MDET] Pre process')
    x = transform(image_rgb)

    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    _, _, H, W = x.shape
    resize = H != input_size or W != input_size

    if resize:
        x = torch.nn.functional.interpolate(
            x,
            size=(input_size, input_size),
            mode=interpolation_mode,
            align_corners=False,
        )

    print(f'[MDET] model input size : {x.shape}') # (1536, 1536)
    # ===================================================================
    print('[MDET] Run inference')
    with torch.no_grad():
        canonical_inverse_depth, fov_deg = model(x)
    # ===================================================================
    print('[MDET] Post process')
    if f_px0 is None:
        f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
    else :
        f_px = f_px0

    inverse_depth = canonical_inverse_depth * (W / f_px)
    f_px = f_px.squeeze()

    if resize:
        inverse_depth = torch.nn.functional.interpolate(
            inverse_depth, size=(H, W), mode=interpolation_mode, align_corners=False
        )

    depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
    # ===================================================================

    # Extract the depth and focal length.
    depth = torch.squeeze(depth).detach().cpu().numpy()
    print(f'[MDET] max : {depth.max():0.2f} , min : {depth.min():0.2f}')
    f_px = f_px.detach().cpu().item()
    if f_px0 is not None:
        print(f"[MDET] focal length (from exif): {f_px:0.2f}")
    else :
        print(f"[MDET] predicted Focal length (by Depth Pro) : {f_px:0.2f}")


    print('[MDET] Generate color depth image')
    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)

    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)
    
    # save colored depth image 
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_torch.jpg')
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0]+ f'_torch')
    np.savez_compressed(output_file_npz, depth=depth)

    print(f'[MDET] see results ({save_dir_path})')

    infer_performace(model)


if __name__ == "__main__":
    main()