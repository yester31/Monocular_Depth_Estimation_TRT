# by yhpark 2025-8-1
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms as TF

import sys
sys.path.insert(1, os.path.join(sys.path[0], "Prior-Depth-Anything"))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from PIL import Image
from prior_depth_anything.plugin import PriorDARefiner, PriorDARefinerMetrics
import open3d as o3d
from enhance_depth import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] Using device: {DEVICE}")
#dtype = torch.half
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

    print(f'[MDET] {iteration} iterations time ({input_h, input_w}): {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

def depth_to_image(depth, width, height, output_path):
    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu + 1e-6)

    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    color_depth_bgr = cv2.resize(color_depth_bgr, (width, height), cv2.INTER_LINEAR)

    # save colored depth image 
    cv2.imwrite(output_path, color_depth_bgr)

def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

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
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    print('[MDET] Load model & image')
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE).eval()
    Refiner = PriorDARefiner(device=DEVICE).eval()

    # input 
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    height, width = raw_image.shape[:2]
    image_names = [image_path] 
    images = load_and_preprocess_images(image_names)
    # images, original_coords = load_and_preprocess_images_square(image_names)
    images = images.to(DEVICE)

    # Reload RGB for refiner.
    rgb_image = Image.open(image_names[0])
    resized_img = rgb_image.resize((images.shape[3],images.shape[2]), Image.BILINEAR)
    priorda_image = torch.from_numpy(np.asarray(resized_img).astype(np.uint8))
    print(f'[MDET] model input size : {images.shape[1:]}') 
    # ===================================================================
    print('[MDET] Run inference')
    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = vggt.aggregator(images)

        # Predict Cameras
        pose_enc = vggt.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = vggt.depth_head(aggregated_tokens_list, images, ps_idx)
        # Predict Point Maps
        point_map, point_conf = vggt.point_head(aggregated_tokens_list, images, ps_idx)
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))
        print('[MDET] Refine depth')
        ### Refine depth 
        # depth_map, depth_conf = predictions['depth'], predictions['depth_conf']
        refined_depth, meview_depth_map = Refiner.predict(image=priorda_image, depth_map=depth_map.squeeze(), confidence=depth_conf.squeeze())
        # The size of `refined_depth` is the same as `priorda_mage`, tune it to your need.

        ### Refine point_map.
        depth_by_project = project_point_map_to_depth_map(point_map.view(-1, 3).unsqueeze(0), extrinsics_cam=extrinsic.squeeze(0), intrinsics_cam=intrinsic.squeeze(0), size=images.shape[-2:])
        refined_projected, meview_depth_by_project = Refiner.predict(image=priorda_image, depth_map=depth_by_project.squeeze(), confidence=point_conf.squeeze())
        inview_refined_projected = F.interpolate(refined_projected[None, None, ...], size=(depth_map.shape[-3], depth_map.shape[-2]), mode='bilinear', align_corners=True).squeeze()
        refined_world_points = unproject_depth_map_to_point_map(inview_refined_projected[None, ..., None], extrinsic.squeeze(0), intrinsic.squeeze(0))


    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_vggt = depth_map[0,...,0]
    refined_depth = refined_depth.squeeze(0).cpu().numpy()
    # ===================================================================
    print(f'[MDET] origianl max : {depth_vggt.max():0.5f} , min : {depth_vggt.min():0.5f}')
    print(f'[MDET] refined max : {refined_depth.max():0.5f} , min : {refined_depth.min():0.5f}')

    print('[MDET] Generate color depth image')
    output_path = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_vggt_torch.jpg')
    depth_to_image(depth_vggt, width, height, output_path)

    output_path = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_vggt_refined_torch.jpg')
    depth_to_image(refined_depth, width, height, output_path)


    points2 = np.stack(refined_world_points, axis=-1).reshape(-1, 3)
    points = np.stack(point_map_by_unprojection, axis=-1).reshape(-1, 3)
    colors = np.array(resized_img).reshape(-1, 3) / 255.0
    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + "_vggt.ply"), pcd)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + "_vggt2.ply"), pcd)

    o3d.visualization.draw_geometries([pcd, pcd2])

    # infer_performace(model, input_h, input_w)

    
if __name__ == "__main__":
    main()