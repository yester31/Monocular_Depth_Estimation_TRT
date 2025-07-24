# by yhpark 2025-7-24
import os
from pathlib import Path
from typing import *
import json
import cv2
import numpy as np
import torch
import matplotlib
import time 

from MoGe.moge.model.v2 import MoGeModel
from MoGe.moge.utils.vis import colorize_depth, colorize_normal
import utils3d

cur_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def infer_performace(model, input_size=(518, 518)):
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device).half()
    num_tokens = torch.tensor(3600).to(device).half()

    # 예열 단계 (GPU 워밍업)
    with torch.no_grad():
        for _ in range(20):
            pred_disp = model(dummy_input, num_tokens)
    torch.cuda.synchronize()

    # FPS 측정
    iteration = 100
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            pred_disp = model(dummy_input, num_tokens)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f'{iteration} iterations time ({input_size}): {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'Average inference time: {avg_time * 1000:.2f} [msec]')


def main():  

    input_path = os.path.join(cur_dir, '..', 'data', 'example.jpg')
    file_name = os.path.splitext(os.path.basename(input_path))[0]

    use_fp16 = True
    resize_to = None
    fov_x_ = None
    resolution_level = 9
    resolution_level = 9/4
    num_tokens = None
    output_path = Path(cur_dir, 'results')

    # checkpoint = "Ruicheng/moge-2-vitl-normal"
    encoder = 'vits' # 'vitl' or 'vitb', 'vits'
    checkpoint = f"{cur_dir}/MoGe/checkpoint/moge-2-{encoder}-normal/model.pt"
    #checkpoint = f"{cur_dir}/MoGe/checkpoint/moge-2-vitl/model.pt"
    model = MoGeModel.from_pretrained(checkpoint).to(device).eval()
    if use_fp16:
        model.half()

    image = cv2.imread(input_path)
    image = cv2.resize(image, (518, 518))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]
    if resize_to is not None:
        height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
        image = cv2.resize(image, (width, height), cv2.INTER_AREA)
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)


    # Inference
    output = model.infer(image_tensor, fov_x=fov_x_, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=use_fp16)
    points = output['points'].cpu().numpy()
    depth = output['depth'].cpu().numpy()
    mask = output['mask'].cpu().numpy()
    intrinsics = output['intrinsics'].cpu().numpy()
    normal = output['normal'].cpu().numpy() if 'normal' in output else None

    # Save
    save_path = Path(output_path, encoder, file_name)
    save_path.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(save_path / 'depth_vis.png'), cv2.cvtColor(colorize_depth(depth=depth), cv2.COLOR_RGB2BGR))
    
    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
    with open(save_path / 'fov.json', 'w') as f:
        json.dump({'fov_x': round(float(np.rad2deg(fov_x)), 2), 'fov_y': round(float(np.rad2deg(fov_y)), 2),}, f)

    cv2.imwrite(str(save_path / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))


    # visualization
    print(f'max : {depth.max()} , min : {depth.min()}')
    depth = 1 / depth
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    cmap = matplotlib.colormaps.get_cmap('turbo')
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, f'{file_name}_moge2_torch.png'), depth)

    infer_performace(model, (height, width))


if __name__ == '__main__':
    main()
