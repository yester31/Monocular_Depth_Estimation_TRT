# by yhpark 2025-7-28
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import contextlib

import torch
import torch.nn.functional as F

import sys
sys.path.insert(1, os.path.join(sys.path[0], "DepthAnythingAC"))
from DepthAnythingAC.depth_anything.dpt import DepthAnything_AC

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Repo root, found by walking up to the directory holding core/ rather than
# counting "..". Counting breaks the moment a script moves, and this one
# has moved: Phase 4 put every model under models/.
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "core")) and os.path.dirname(_R) != _R:
    _R = os.path.dirname(_R)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")


@contextlib.contextmanager
def upstream_cwd():
    """Construct the model with DepthAnythingAC/ as the working directory.

    Upstream's `dpt.py` builds the DINOv2 backbone through its vendored
    torchhub, which loads the weights from the *relative* path
    `checkpoints/dinov2_vits14/dinov2_vits14_pretrain.pth`. So the model can
    only be built from that directory, no matter where the script lives.

    infer.py used to work around this by copying the entire checkpoints tree
    up one level (copy_checkpoints below) before building. onnx_export.py
    calls set_model() without that side effect, so it died with
    FileNotFoundError on the dinov2 weights. Setting the directory for the
    duration of the call is cheaper than duplicating 100 MB of weights and
    works from any caller.
    """
    prev = os.getcwd()
    os.chdir(os.path.join(CUR_DIR, 'DepthAnythingAC'))
    try:
        yield
    finally:
        os.chdir(prev)


# copy_checkpoints() lived here. It duplicated the whole DepthAnythingAC/
# checkpoints tree one level up so the relative DINOv2 path would resolve with
# this directory as the working directory. upstream_cwd() does the same job
# without copying 100 MB, and works for onnx_export.py too, which never called
# it. Removed along with main()'s call to it.


def infer_performace(model, input_size=518):

    dummy_input = torch.randn(1, 3, input_size, input_size).to(DEVICE).half()

    # 예열 단계 (GPU 워밍업)
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # FPS 측정
    iteration = 100
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f'[MDET] {iteration} iterations time ({input_size, input_size}): {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

def preprocess_image(image, target_size=518):
    """Preprocess input image"""
    
    h, w = image.shape[:2]
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    new_h = ((new_h + 13) // 14) * 14
    new_w = ((new_w + 13) // 14) * 14
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    
    return image

def set_model(encoder='vits', dtype: torch.dtype = torch.float32):
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'version': 'v2'},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'version': 'v2'},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'version': 'v2'}
    }
 
    # Absolute path, and it points into the upstream clone, which is where the
    # README's huggingface-cli download puts the weights. The wrapper handles
    # the DINOv2 backbone, whose path upstream resolves against the working
    # directory.
    with upstream_cwd():
        model = DepthAnything_AC(model_configs[encoder])
        checkpoint = torch.load(
            f'{CUR_DIR}/DepthAnythingAC/checkpoints/depth_anything_AC_{encoder}.pth',
            map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.to(DEVICE).eval()

    if dtype == torch.half:
        model = model.half()
    
    return model

def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    print('[MDET] Load model & image')
    encoder = 'vits' # 'vits' 
    dtype = torch.half
    model = set_model(encoder, dtype)

    image_file_name = 'example.jpg'
    image_path = os.path.join(_R, 'data', image_file_name)
    raw_image = cv2.imread(image_path) # Load image.
    # raw_image = cv2.resize(raw_image, (518, 518))
    # ===================================================================
    print('[MDET] Pre process')
    ori_shape = raw_image.shape[:2]
    print(f'[MDET] original image size : {ori_shape}') # 
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = preprocess_image(image)
    x = image.unsqueeze(0).to(DEVICE)
    if dtype == torch.half:
        x = x.half()
    print(f'[MDET] model input size : {x.shape}')
    # ===================================================================
    print('[MDET] Run inference')
    with torch.no_grad():
        prediction = model(x)
        # ===================================================================
        print('[MDET] Post process')
        depth = prediction['out']
        depth = F.interpolate(depth[:, None], ori_shape, mode="bilinear", align_corners=True)[0, 0]
        depth = torch.clamp(depth, min=1e-3, max=1e3)
        depth = torch.squeeze(depth).detach().cpu().numpy()

    print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')

    # ===================================================================
    print('[MDET] Generate color depth image')
    # visualization
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    

    # save colored depth image 
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_torch.jpg')
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0]+ f'_torch')
    np.savez_compressed(output_file_npz, depth=depth)

    infer_performace(model)


if __name__ == "__main__":
    main()