# by yhpark 2025-8-1
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import cv2

import torch
import torch.nn.functional as F

from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map

import open3d as o3d

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
#dtype = torch.half
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(f"Using dtype: {dtype}")

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

    print(f'[MDET] {iteration} iterations time ({input_h, input_w}): {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    print('[MDET] Load model & image')
    input_h, input_w = 518, 518 # 518, 518
    target_size = 1024
    original_coords = []  # Renamed from position_info to be more descriptive

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(DEVICE)

    # input 
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    # ===================================================================
    # original -> 1024 -> 518 - > 1024 -> original
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
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        print(f'[MDET] pose_enc : \n{pose_enc.reshape((3,3))}')
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        # Predict Point Maps
        point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        depth_map = depth_map.squeeze(0) # [1,518,518,1]
        depth_map = depth_map.permute(0, 3, 1, 2) # [1, 1, 518, 518]
        depth_map = F.interpolate(depth_map, size=(target_size, target_size), mode="bilinear", align_corners=False) # [1, 1, 1024, 1024]
        depth_map = depth_map[...,int(y1):int(y2), int(x1):int(x2)] # remove paddings # [1, 1, 768, 1024]
        depth_map = depth_map.permute(0, 2, 3, 1) # [1, 768, 1024, 1]
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map, extrinsic.squeeze(0), intrinsic.squeeze(0))

    #extrinsic = extrinsic.squeeze(0).cpu().numpy()
    #intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    #depth_conf = depth_conf.squeeze(0).cpu().numpy()
    # ===================================================================
    print('[MDET] Post process')
    depth = np.squeeze(depth_map)
    # depth = cv2.resize(depth, (target_size, target_size))
    # original_coord = original_coords[0]
    # depth = depth[int(original_coord[1]) : int(original_coord[3]), int(original_coord[0]):int(original_coord[2])] # remove paddings
    # depth = depth[int(y1):int(y2), int(x1):int(x2)] # remove paddings
    input_h2, input_w2 = depth.shape
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
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_vggt_torch.jpg')
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save point cloud 
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) 
    rgb_resized_img = cv2.resize(rgb_image, (input_w2, input_h2), cv2.INTER_LINEAR)
    points = np.stack(point_map_by_unprojection, axis=-1).reshape(-1, 3)
    colors = np.array(rgb_resized_img).reshape(-1, 3) / 255.0
    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + "_vggt.ply"), pcd)
    o3d.visualization.draw_geometries([pcd])

    infer_performace(model, input_h, input_w)

    
if __name__ == "__main__":
    main()