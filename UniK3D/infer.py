# by yhpark 2025-7-31
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import json
import torch
from UniK3D.unik3d.models import UniK3D
from safetensors.torch import load_file 

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")


def infer_performace(model, input_h=518, input_w=518):
    dummy_input = torch.randn(1, 3, input_h, input_w).to(DEVICE).half()

    with torch.no_grad():
        for _ in range(20):
            _ = model.infer(dummy_input)
    torch.cuda.synchronize()

    iteration = 100
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            _ = model.infer(dummy_input)
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
    input_h, input_w = 518, 518 # 518, 518
    dtype = torch.half
    encoder = 'vitb' # 'vits' or 'vitb' or 'vitl'
    print(f'[MDET] backbone : {encoder}')
    if 1 :
        with open(f"{CUR_DIR}/UniK3D/configs/eval/{encoder}.json") as f:
            config = json.load(f)
        model = UniK3D(config)
        checkpoint_path = os.path.join(CUR_DIR, "UniK3D","checkpoints", encoder, "model.safetensors")
        model_weights = load_file(checkpoint_path)
        model.load_state_dict(model_weights)
    else:
        model = UniK3D.from_pretrained(f"lpiccinelli/unik3d-{encoder}") 

    model.eval().to(DEVICE)
    if dtype == torch.half:
        model = model.half()

    # input 
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    #raw_image = cv2.resize(raw_image, (518, 518))
    # ===================================================================
    print('[MDET] Pre process')
    ori_shape = raw_image.shape[:2]
    print(f"[MDET] original image size : {ori_shape}")
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) 
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    if dtype == torch.half:
        x = x.half()
    print(f'[MDET] model input size : {x.shape}') # 
    # ===================================================================
    print('[MDET] Run inference')
    with torch.no_grad():
        predictions = model.infer(x)
    # ===================================================================
    print('[MDET] Post process')
    # Point Cloud in Camera Coordinate
    xyz = predictions["points"]
    # Unprojected rays
    rays = predictions["rays"]
    # Metric Depth Estimation
    depth = predictions["depth"]

    depth = torch.squeeze(depth).detach().cpu().numpy()
    print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')
    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu + 1e-6)

    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    color_depth_bgr = cv2.resize(color_depth_bgr, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)

    # save colored depth image 
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{encoder}_unik3d_torch.jpg')
    cv2.imwrite(output_file_depth, color_depth_bgr)

    infer_performace(model, input_h, input_w)

    
if __name__ == "__main__":
    main()