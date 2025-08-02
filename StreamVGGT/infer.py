# by yhpark 2025-8-2
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

import torch
import torch.nn.functional as F

import sys
sys.path.insert(1, os.path.join(sys.path[0], "StreamVGGT/src"))
from StreamVGGT.src.streamvggt.models.streamvggt import StreamVGGT
from StreamVGGT.src.streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(f"[MDET] Using dtype: {dtype}")

def infer_performace(model, input_h=518, input_w=518):
    dummy_input = torch.randn(1, 1, 3, input_h, input_w).to(DEVICE)

    with torch.no_grad():
        for _ in range(3):
            with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype):
                aggregated_tokens_list, ps_idx = model.aggregator(dummy_input)
            # Predict Cameras
            # pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, dummy_input.shape[-2:])
            # Predict Depth Maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, dummy_input, ps_idx)
    torch.cuda.synchronize()

    iteration = 10
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype):
                aggregated_tokens_list, ps_idx = model.aggregator(dummy_input)
            # Predict Cameras
            # pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, dummy_input.shape[-2:])
            # Predict Depth Maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, dummy_input, ps_idx)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    dur_time_str = f"{dur_time:.2f} [sec]" if dur_time < 60 else f"{dur_time // 60 :.1f} [min] {dur_time % 60 :.2f} [sec]"
    print(f'[MDET] input size : {input_h}, {input_w}')
    print(f'[MDET] {iteration} iterations time : {dur_time_str}')
    avg_time = dur_time / iteration
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average Latency: {avg_time * 1000:.2f} [msec]')

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    print('[MDET] Load model & image')
    input_h, input_w = 518, 518 # 518, 518
    target_size = 1024
    original_coords = []  # Renamed from position_info to be more descriptive

    model = StreamVGGT()
    weights = f"{CUR_DIR}/StreamVGGT/ckpt/checkpoints.pth"
    ckpt = torch.load(weights, map_location="cuda")
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model = model.to(DEVICE)
    print(f"Model loaded")

    # input 
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    # ===================================================================
    print('[MDET] Pre process')
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height, width}")
    img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) 

    # Make the image square by padding the shorter dimension
    max_dim = max(width, height)

    # Calculate padding
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2

    # Calculate scale factor for resizing
    scale = target_size / max_dim

    # Calculate final coordinates of original image in target space
    x1 = left * scale
    y1 = top * scale
    x2 = (left + width) * scale
    y2 = (top + height) * scale

    # Store original image coordinates and scale
    original_coords.append(np.array([x1, y1, x2, y2, width, height]))

    # Create a new black square image and paste original
    padding = [0, 0, 0]
    img = cv2.copyMakeBorder(img, top, top, left, left, cv2.BORDER_CONSTANT, value=padding)

    # Resize to target size
    rgb = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    # Convert to tensor
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float() / 255.0
    rgb = rgb.unsqueeze(0).to(DEVICE)
    images = F.interpolate(rgb, size=(input_h, input_w), mode="bilinear", align_corners=False)

    print(f'[MDET] model input size : {images.shape}') # [1, 3, 518, 518]
    # ===================================================================
    print('[MDET] Run inference')
    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        # pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    #extrinsic = extrinsic.squeeze(0).cpu().numpy()
    #intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    #depth_conf = depth_conf.squeeze(0).cpu().numpy()
    # ===================================================================
    print('[MDET] Post process')
    depth = depth_map[0,...,0]

    original_coord = original_coords[0]
    depth = depth[int(original_coord[1]/2) : int(original_coord[3]/2), :] # remove paddings
    depth = cv2.resize(depth, (int(original_coord[4]), int(original_coord[5])))
    print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')

    print('[MDET] Generate color depth image')
    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu + 1e-6)

    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    color_depth_bgr = cv2.resize(color_depth_bgr, (width, height), cv2.INTER_LINEAR)

    # save colored depth image 
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_svggt_torch.jpg')
    cv2.imwrite(output_file_depth, color_depth_bgr)

    infer_performace(model, input_h, input_w)

    
if __name__ == "__main__":
    main()