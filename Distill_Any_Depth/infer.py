import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

import torch
from torchvision.transforms import Compose

from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from safetensors.torch import load_file 

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

def set_model(encoder='vits', input_h=518, input_w=518, dtype: torch.dtype = torch.float32):
    checkpoint_path = f'{CUR_DIR}/Distill-Any-Depth/checkpoint/{encoder}/model.safetensors'       
    arch_name = f"depthanything-{encoder}" # 'depthanything-large', 'depthanything-base', 'depthanything-small'

    if encoder == 'Large-2w-iter':
        arch_name = f"depthanything-large" # 'depthanything-large', 'depthanything-base', 'depthanything-small'

    model_kwargs = dict(
        vits=dict(
            encoder='vits', 
            features=64,
            out_channels=[48, 96, 192, 384]
        ),
        vitb=dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        vitl=dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        )
    )

    # Load model
    if arch_name == 'depthanything-large':
        model = DepthAnything(**model_kwargs['vitl']).to(DEVICE)
    elif arch_name == 'depthanything-base':
        model = DepthAnythingV2(**model_kwargs['vitb']).to(DEVICE)
    elif arch_name == 'depthanything-small':
        model = DepthAnythingV2(**model_kwargs['vits']).to(DEVICE)
    else:
        raise NotImplementedError(f"Unknown architecture: {arch_name}")
    
    # safetensors 
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    del model_weights
    torch.cuda.empty_cache()
    model = model.eval()

    if dtype == torch.half:
        model = model.half()

    transform = Compose([
        Resize(input_w, 
               input_h, 
               resize_target=False, 
               keep_aspect_ratio=False, 
               ensure_multiple_of=14, 
               resize_method='lower_bound', 
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

    return model, transform

def infer_performace(model, input_h=518, input_w=518):
    dummy_input = torch.randn(1, 3, input_h, input_w).to(DEVICE).half()

    with torch.no_grad():
        for _ in range(20):
            pred_disp, _ = model(dummy_input)
    torch.cuda.synchronize()

    iteration = 100
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            pred_disp, _ = model(dummy_input)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f'[MDET] {iteration} iterations time ({input_h, input_w}): {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    print('[MDET] Load model & image')
    encoder = 'small' # 'large' or 'base' or 'small' or 'Large-2w-iter'
    input_h, input_w = 518, 518 # 518, 518
    dtype = torch.half
    model, transform = set_model(encoder, input_h, input_w, dtype)

    # input 
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    #raw_image = cv2.resize(raw_image, (518, 518))
    # ===================================================================
    print('[MDET] Pre process')
    ori_shape = raw_image.shape[:2]
    print(f"[MDET] original image size : {ori_shape}")
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({'image': image})['image']
    x = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    if dtype == torch.half:
        x = x.half()
    print(f'[MDET] model input size : {x.shape}') # 
    # ===================================================================
    print('[MDET] Run inference')
    with torch.no_grad():
        pred_disp, _ = model(x)
    # ===================================================================
    print('[MDET] Post process')
    depth = torch.squeeze(pred_disp).detach().cpu().numpy()
    print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)

    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    color_depth_bgr = cv2.resize(color_depth_bgr, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)

    # save colored depth image 
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{encoder}_dad_torch.jpg')
    cv2.imwrite(output_file_depth, color_depth_bgr)

    infer_performace(model, input_h, input_w)

    
if __name__ == "__main__":
    main()