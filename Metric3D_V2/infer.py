# by yhpark 2025-7-30
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

import torch
import torch.nn.functional as F

import sys
sys.path.insert(1, os.path.join(sys.path[0], "Metric3D"))
from Metric3D.hubconf import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

def infer_performace(model, input_sizes):

    dummy_input = torch.randn(1, 3, input_sizes[0], input_sizes[1]).to(DEVICE)

    # 예열 단계 (GPU 워밍업)
    with torch.no_grad():
        for _ in range(20):
            _, _, _ = model.inference({'input': dummy_input})
    torch.cuda.synchronize()

    # FPS 측정
    iteration = 100
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            _, _, _ = model.inference({'input': dummy_input})
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f'[MDET] {iteration} iterations time ({input_sizes[0], input_sizes[1]}): {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    print('[MDET] Load model & image')
    encoder = 'vitl' # 'vits' or vitl or vitg
    #model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    if encoder == 'vits':
        model = metric3d_vit_small(pretrain=True)
    elif encoder == 'vitl':
        model = metric3d_vit_large(pretrain=True)
    elif encoder == 'vitg':
        model = metric3d_vit_giant2(pretrain=True)
    #model = metric3d_convnext_tiny(pretrain=True)
    #model = metric3d_convnext_large(pretrain=True)
    model.eval().to(DEVICE)

    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path) # Load image.
    # ===================================================================
    print('[MDET] Pre process')
    ori_shape = raw_image.shape[:2]
    print(f'[MDET] original image size : {ori_shape}') # 
    rgb_origin = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    #### ajust input size to fit pretrained model
    # keep ratio resize
    input_sizes = (616, 1064) # for vit model
    # input_sizes = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_sizes[0] / h, input_sizes[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic, hold depth
    intrinsic = [707.0493, 707.0493, 604.0814, 180.5066] # from Metric3D/hubconf.py
    # intrinsic = [872, 1163, 500, 500] # from moge2
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_sizes[0] - h
    pad_w = input_sizes[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].to(DEVICE)

    print(f'[MDET] model input size : {rgb.shape}')
    # ===================================================================
    print('[MDET] Run inference')
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})    
    # ===================================================================
    print('[MDET] Post process')
    # Metric Depth Estimation
    # un pad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
    ###################### canonical camera space ######################

    #### de-canonical transform
    if 0 :
        real_focal_length = intrinsic[0]
        real_focal_length = 1440  
        real_focal_length = 2890  
        real_focal_length = 3365.20 # from depth pro  

        print(f'f_length : {real_focal_length}, scale : {scale}, f_length * scale : {real_focal_length * scale}')
        canonical_focal_length = 1000.0 # 1000.0 is the focal length of canonical camera
        canonical_to_real_scale = real_focal_length * scale / canonical_focal_length 
        pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)

    #### normal are also available
    if 'prediction_normal' in output_dict: # only available for Metric3Dv2, i.e. vit model
        pred_normal = output_dict['prediction_normal'][:, :3, :, :]
        normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
        # un pad and resize to some size if needed
        pred_normal = pred_normal.squeeze()
        pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
        # you can now do anything with the normal
        # such as visualize pred_normal
        pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
        pred_normal_vis = (pred_normal_vis + 1) / 2
        cv2.imwrite(f'{save_dir_path}/{os.path.splitext(image_file_name)[0]}_{encoder}_normal_vis.png', (pred_normal_vis * 255).astype(np.uint8))


    depth = pred_depth.cpu()
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

    infer_performace(model, input_sizes)


if __name__ == "__main__":
    main()