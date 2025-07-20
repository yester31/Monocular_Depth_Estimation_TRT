import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path as osp
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision.transforms import Compose
import time

sys.path.insert(1, os.path.join(sys.path[0], "Distill-Any-Depth"))

from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from safetensors.torch import load_file 

cur_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Helper function for model loading
def load_model_by_name(arch_name, checkpoint_path, device):
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
        model = DepthAnything(**model_kwargs['vitl']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"large/model.safetensors", repo_type="model")

    elif arch_name == 'depthanything-base':
        model = DepthAnythingV2(**model_kwargs['vitb']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"base/model.safetensors", repo_type="model")
    elif arch_name == 'depthanything-small':
        model = DepthAnythingV2(**model_kwargs['vits']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"base/model.safetensors", repo_type="model")

    else:
        raise NotImplementedError(f"Unknown architecture: {arch_name}")
    
    # safetensors 
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    del model_weights
    torch.cuda.empty_cache()
    return model


def infer_performace(model, input_size=700):
    model = model.half()
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device).half()

    # 예열 단계 (GPU 워밍업)
    with torch.no_grad():
        for _ in range(20):
            pred_disp, _ = model(dummy_input)
    torch.cuda.synchronize()

    # FPS 측정
    iteration = 100
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            pred_disp, _ = model(dummy_input)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f'{iteration} iterations time: {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'Average inference time: {avg_time * 1000:.2f} [msec]')


def main():

    # Model preparation
    encoder = 'small' # 'large' or 'base', 'small'
    checkpoint = f'{cur_dir}/Distill-Any-Depth/checkpoint/{encoder}/model.safetensors'
    arch_name = f"depthanything-{encoder}" # 'depthanything-large', 'depthanything-base', 'depthanything-small'
    model = load_model_by_name(arch_name, checkpoint, device)

    # Define image transformation
    # resize_h, resize_w = 700, 700
    resize_h, resize_w = 518, 518
    transform = Compose([
        Resize(resize_w, resize_h, resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

    # input 
    image_path = os.path.join(cur_dir, '..', 'data', 'example.jpg')
    validation_image_np = cv2.imread(image_path, cv2.COLOR_BGR2RGB)[..., ::-1] / 255
    print(f"original input size : {validation_image_np.shape}")
    validation_image = transform({'image': validation_image_np})['image']
    validation_image = torch.from_numpy(validation_image).unsqueeze(0).to(device).half()
    model = model.eval().half()
    # infer
    with torch.autocast("cuda"):
        pred_disp, _ = model(validation_image)
    
    # post-proc
    pred_disp_np = pred_disp.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0)
    pred_disp = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())

    # visualization
    cmap = "turbo" # "turbo" or "Spectral_r"
    depth_colored = colorize_depth_maps(pred_disp[None, ...], 0, 1, cmap=cmap).squeeze()
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    h, w = validation_image_np.shape[:2]
    depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)
    image_out = Image.fromarray(np.concatenate([depth_colored_hwc], axis=1))

    # save result
    outdir = 'results'
    os.makedirs(outdir, exist_ok=True)
    image_out.save(osp.join(cur_dir, outdir, f'{os.path.splitext(os.path.basename(image_path))[0]}_{encoder}_dad_torch.png'))
    torch.cuda.empty_cache()

    infer_performace(model, resize_h)

    
if __name__ == "__main__":
    main()