# by yhpark 2025-7-16
# check Depth Pro pytorch model inference performance
import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(f'{cur_dir}/ml-depth-pro'))
depth_pro = __import__('depth_pro')

import torch
import numpy as np
import shutil
import time 

from PIL import Image
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def copy_checkpoints():
    src = f'{cur_dir}/ml-depth-pro/checkpoints'
    dst = f'{cur_dir}/checkpoints'

    try:
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            print("복사 완료")
        else:
            print("이미 존재함, 복사하지 않음")
    except FileNotFoundError:
        print("복사할 원본 폴더가 존재하지 않음")
    except Exception as e:
        print("오류 발생:", e)

def post_process(depth):
    inverse_depth = 1 / depth
    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )
    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    return color_depth

def save_color_depth(color_depth, image_file_name):
    # 샘플 결과 출력 및 저장
    save_path = os.path.join(cur_dir, 'results', f'{os.path.splitext(image_file_name)[0]}_Torch.jpg')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(color_depth).save(save_path, format="JPEG", quality=90)

def save_npz(depth, image_file_name):
    output_file_npz = os.path.join(cur_dir, 'results', os.path.splitext(image_file_name)[0])
    np.savez_compressed(output_file_npz, depth=depth)

def infer_one_vis(model, transform, image_file_name='example.jpg'):

    # Load image.
    image_path = os.path.join(cur_dir, 'ml-depth-pro', 'data', image_file_name)
    image0, _, f_px = depth_pro.load_rgb(image_path) # RGB

    # pre-process
    image = transform(image0)

    # Run inference
    prediction = model.infer(image, f_px=f_px)

    # post-process

    # Extract the depth and focal length.
    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    if f_px is not None:
        print(f"Focal length (from exif): {f_px:0.2f}")
    elif prediction["focallength_px"] is not None:
        focallength_px = prediction["focallength_px"].detach().cpu().item()
        print(f"Estimated focal length: {focallength_px}")

    # generate color depth image
    color_depth = post_process(depth)
    save_color_depth(color_depth, image_file_name)
    save_npz(depth, image_file_name)
    print('color depth image done (.results/example_Torch.jpg)')

def infer_performace(model):

    dummy_input = torch.randn(1, 3, model.img_size, model.img_size).to(device).half()

    # 예열 단계 (GPU 워밍업)
    with torch.no_grad():
        for _ in range(5):
            _ = model.infer(dummy_input)
    torch.cuda.synchronize()

    # FPS 측정
    iteration = 20
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            _ = model.infer(dummy_input)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f'{iteration} iterations time: {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'Average inference time: {avg_time * 1000:.2f} [msec]')


def main():
    copy_checkpoints()

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms(device=device, precision=torch.half)
    model.eval()
    print(f'model input size : {(model.img_size, model.img_size)}') # (1536, 1536)

    infer_one_vis(model, transform)

    infer_performace(model)


if __name__ == "__main__":
    main()