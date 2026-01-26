# by yhpark 2026-1-23
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

import torch
import torchvision.transforms as T

from depth_anything_3.api import DepthAnything3

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

def infer_performace(model, input_size=518):

    dummy_input = torch.randn(1, 1, 3, input_size, input_size).to(DEVICE).half()

    # 예열 단계 (GPU 워밍업)
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input, None, None, [])
    torch.cuda.synchronize()

    # FPS 측정
    iteration = 100
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            _ = model(dummy_input, None, None, [])
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f'[MDET] {iteration} iterations time ({input_size, input_size}): {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    print('[MDET] Load model & image')
    input_size = 518
    model = DepthAnything3.from_pretrained(f"{CUR_DIR}/Depth_Anything_V3/depth-anything/DA3METRIC-LARGE")
    model = model.eval().to(device=DEVICE)
    transform = T.Compose(
        [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path) # Load image.
    # ===================================================================
    print('[MDET] Pre process')
    height, width = raw_image.shape[:2]
    print(f'[MDET] original image size : {height, width}') 

    raw_image = cv2.resize(raw_image, (input_size, input_size))
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    x = transform(image).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    print(f'[MDET] model input size : {x.shape}') #
    # ===================================================================
    print('[MDET] Run inference')
    with torch.no_grad():
        predictions = model(x, None, None, [])
    # ===================================================================
    print('[MDET] Post process')
    depth = predictions['depth']
    sky = predictions['sky']
    depth = depth[0, 0]
    depth = torch.clamp(depth, min=1e-3, max=1e3)
    depth = torch.squeeze(depth).detach().cpu().numpy()
    print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')

    # ===================================================================
    print('[MDET] Generate color depth image')
    # visualization
    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu + 1e-6)

    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    color_depth_bgr = cv2.resize(color_depth_bgr, (width, height), cv2.INTER_LINEAR)

    # save colored depth image 
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_dav3_torch.jpg')
    cv2.imwrite(output_file_depth, color_depth_bgr)

    infer_performace(model)


if __name__ == "__main__":
    main()