# by yhpark 2025-7-17
import cv2
import torch
import matplotlib
import numpy as np
import time
import os
import sys

from torchvision.transforms import Compose
import torch.nn.functional as F

sys.path.insert(1, os.path.join(sys.path[0], "Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

cur_dir = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def post_proc(depth, ori_shape):
    depth = F.interpolate(depth[:, None], ori_shape, mode="bilinear", align_corners=True)[0, 0]
        
    return depth.cpu().numpy()

def image2tensor(raw_image, input_size=518):        
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    h, w = raw_image.shape[:2]
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0)
    image = image.to(DEVICE)
    
    return image, (h, w)

def save_npz(depth, image_file_name):
    output_file_npz = os.path.join(cur_dir, 'results', image_file_name)
    np.savez_compressed(output_file_npz, depth=depth)

def infer_one_vis(model, image_path, encoder):
    # Load image.
    raw_img = cv2.imread(image_path)
    # pre-process
    image, (h, w) = image2tensor(raw_img)
    print(f'after preprocess shape : {image.shape}')

    with torch.no_grad():
        # Run inference
        depth = model(image.half())
        print(f'output shape : {depth.shape}')
        # post-process
        depth0 = post_proc(depth, (h, w))

    # visualization
    depth = (depth0 - depth0.min()) / (depth0.max() - depth0.min()) * 255.0
    depth = depth.astype(np.uint8)
    #cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    cmap = matplotlib.colormaps.get_cmap('turbo')
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    outdir = 'results'
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, os.path.splitext(os.path.basename(image_path))[0] + f'_{encoder}_Torch.png'), depth)

    save_npz(depth0, os.path.splitext(os.path.basename(image_path))[0] + f'_{encoder}')
    print(f'color depth image done (.results/example_{encoder}_Torch.png)')

def infer_performace(model, input_size=518):

    dummy_input = torch.randn(1, 3, input_size, input_size).to(DEVICE).half()

    # 예열 단계 (GPU 워밍업)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # FPS 측정
    iteration = 20
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f'{iteration} iterations time: {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'Average inference time: {avg_time * 1000:.2f} [msec]')


def main():

    # Load model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vits' # or 'vits', 'vitb', 'vitg'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'./Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval().half()

    image_path = os.path.join(cur_dir, '..', 'data', 'example.jpg')

    infer_one_vis(model, image_path, encoder)

    infer_performace(model)


if __name__ == "__main__":
    main()


# 20 iterations time: 0.5182 [sec]
# Average FPS: 38.60 [fps]
# Average inference time: 25.91 [msec]