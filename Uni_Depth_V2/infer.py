# by yhpark 2025-7-29
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

import torch
import torch.nn.functional as F

from UniDepth.unidepth.models import UniDepthV2

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

def infer_performace(model, input_size=518):

    dummy_input = torch.randn(1, 3, input_size, input_size).to(DEVICE).half()

    # 예열 단계 (GPU 워밍업)
    with torch.no_grad():
        for _ in range(20):
            _ = model.infer(dummy_input)
    torch.cuda.synchronize()

    # FPS 측정
    iteration = 100
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            _ = model.infer(dummy_input)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f'[MDET] {iteration} iterations time ({input_size, input_size}): {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

def set_model(encoder='vits', dtype: torch.dtype = torch.float32):
    model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-{encoder}14") 
    model.to(DEVICE).eval()

    if dtype == torch.half:
        model = model.half()
    
    return model

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    print('[MDET] Load model & image')
    encoder = 'vitb' # 'vits' or vitb or vitl
    dtype = torch.half
    model = set_model(encoder, dtype)

    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path) # Load image.
    # raw_image = cv2.resize(raw_image, (518, 518))
    # ===================================================================
    print('[MDET] Pre process')
    ori_shape = raw_image.shape[:2]
    print(f'[MDET] original image size : {ori_shape}') # 
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = torch.from_numpy(image).permute(2, 0, 1) # C, H, W

    #x = x.unsqueeze(0).to(DEVICE)
    #if dtype == torch.half:
    #    x = x.half()
    print(f'[MDET] model input size : {x.shape}')
    # ===================================================================
    print('[MDET] Run inference')
    with torch.no_grad():
        predictions = model.infer(x)
        # ===================================================================
        print('[MDET] Post process')
        # Metric Depth Estimation
        depth = predictions["depth"] # [1,1,h,w]
        # Point Cloud in Camera Coordinate
        xyz = predictions["points"] # [1,3,h,w]
        # Intrinsics Prediction
        intrinsics = predictions["intrinsics"] # [1,3,3]
        #depth = depth.squeeze(0) # [1,1,h,w] - > [1,h,w] 
        #depth = depth[:, None] # [1,h,w] -> [1,1,h,w]
        depth = F.interpolate(depth, ori_shape, mode="bilinear", align_corners=True)[0, 0]
        depth = torch.clamp(depth, min=1e-3, max=1e3)
        depth = torch.squeeze(depth).detach().cpu().numpy()

    print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')

    # ===================================================================
    print('[MDET] Generate color depth image')
    # visualization
    #depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    inverse_depth = 1 / depth
    #max_invdepth_vizu = np.nanquantile(inverse_depth, 0.99)
    #min_invdepth_vizu = np.nanquantile(inverse_depth, 0.001)
    #print(f'[MDET] max : {1/min_invdepth_vizu:0.5f} , min : {1/max_invdepth_vizu:0.5f}')
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu + 1e-6)

    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    

    # save colored depth image 
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{encoder}_torch.jpg')
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0]+ f'_{encoder}_torch')
    np.savez_compressed(output_file_npz, depth=depth)

    infer_performace(model)


if __name__ == "__main__":
    main()