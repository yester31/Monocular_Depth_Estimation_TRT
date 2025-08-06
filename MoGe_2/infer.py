# by yhpark 2025-7-28
import os
from typing import *
import json
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import time 

from MoGe.moge.model.v2 import MoGeModel
from MoGe.moge.utils.geometry_torch import recover_focal_shift
import utils3d
import trimesh
import trimesh.visual
from PIL import Image

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

def infer_performace(model, input_size=(518, 518)):
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(DEVICE).half()
    num_tokens = torch.tensor(3600).to(DEVICE)

    with torch.no_grad():
        for _ in range(20):
            pred_disp = model(dummy_input, num_tokens)
    torch.cuda.synchronize()

    iteration = 100
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            pred_disp = model(dummy_input, num_tokens)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f'[MDET] {iteration} iterations time ({input_size}): {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

def set_model(encoder='vits', normal=True, dtype: torch.dtype = torch.float32):
    
    if normal:
        checkpoint = f"{CUR_DIR}/MoGe/checkpoint/moge-2-{encoder}-normal/model.pt"
    else:
        encoder='vitl'
        checkpoint = f"{CUR_DIR}/MoGe/checkpoint/moge-2-{encoder}/model.pt"

    model = MoGeModel.from_pretrained(checkpoint).to(DEVICE)
    model = model.eval()

    if dtype == torch.half:
        model.half()

    return model

def main():  

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    print('[MDET] Load model & image')
    encoder = 'vits' # 'vitl' or 'vitb', 'vits'
    normal = True # True or False
    dtype = torch.half # torch.half or torch.float32
    model = set_model(encoder, normal, dtype)

    # input
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    # raw_image = cv2.resize(raw_image, (518, 518))
    # ===================================================================
    print('[MDET] Pre process')
    ori_shape = raw_image.shape[:2]
    print(f"[MDET] original image size : {ori_shape}")
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    resize_to = 518 # 518 or None
    if resize_to is not None:
        height, width = ori_shape[0], ori_shape[1]
        height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
        print(f"[MDET] resize_to : {height, width}")
        image_rgb_resized = cv2.resize(image_rgb, (width, height), cv2.INTER_AREA)

    image = image_rgb_resized / 255.0
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    if dtype == torch.half:
        x = x.half()
    print(f'[MDET] model input size : {x.shape}') 
    num_tokens = 1800 # [1200, 3600]
    print(f'[MDET] num tokens : {num_tokens}')
    # ===================================================================
    print('[MDET] Run inference')
    # Inference
    with torch.no_grad():
        output = model(x, num_tokens)
    # ===================================================================
        print('[MDET] Post process')
        points, normal, mask, metric_scale = (output.get(k, None) for k in ['points', 'normal', 'mask', 'metric_scale'])
        points, normal, mask, metric_scale = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [points, normal, mask, metric_scale])

        if mask is not None:
            mask_binary = mask > 0.5
        else:
            mask_binary = None

        focal, shift = recover_focal_shift(points, mask_binary)

        aspect_ratio = x.shape[3] / x.shape[2]
        fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
        points[..., 2] += shift[..., None, None]
        if mask_binary is not None:
            mask_binary &= points[..., 2] > 0        # in case depth is contains negative values (which should never happen in practice)
        depth = points[..., 2].clone()

        points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)

        # Apply metric scale
        if metric_scale is not None:
            if points is not None:
                points *= metric_scale[:, None, None, None]
            if depth is not None:
                depth *= metric_scale[:, None, None]

        # Apply mask
        if mask_binary is not None:
            points = torch.where(mask_binary[..., None], points, torch.inf) if points is not None else None
            depth = torch.where(mask_binary, depth, torch.inf) if depth is not None else None
            normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal)) if normal is not None else None
        
    points = points.cpu().numpy().squeeze(0)        # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
    depth = depth.cpu().numpy().squeeze(0)          # depth map
    normal = normal.cpu().numpy().squeeze(0)        # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
    mask = mask_binary.cpu().numpy().squeeze(0)     # a binary mask for valid pixels. 
    intrinsics = intrinsics.cpu().numpy().squeeze(0)# normalized camera intrinsics

    save_prefix = os.path.join(save_dir_path, f'{os.path.splitext(image_file_name)[0]}_{encoder}_m2_torch')
    # fov
    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
    with open(f'{save_prefix}_fov.json', 'w') as f:
        json.dump({'fov_x': round(float(np.rad2deg(fov_x).item()), 2), 'fov_y': round(float(np.rad2deg(fov_y).item()), 2),}, f)
    
    # normal
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    color_normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    #color_normal_bgr = cv2.resize(color_normal_bgr, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    cv2.imwrite(f'{save_prefix}_normal.jpg', color_normal_bgr)

    # depth
    print(f'[MDET] max : {depth.max()} , min : {depth.min()}')
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)

    inverse_depth = 1 / depth
    max_invdepth_vizu = np.nanquantile(inverse_depth, 0.99)
    min_invdepth_vizu = np.nanquantile(inverse_depth, 0.001)
    #max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1) # from dpeth pro
    #min_invdepth_vizu = max(1 / 250, inverse_depth.min()) # from dpeth pro
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)

    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    #color_depth_bgr = cv2.resize(color_depth_bgr, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    cv2.imwrite(f'{save_prefix}_depth.jpg', color_depth_bgr)

    depth = cv2.resize(depth, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    normal = cv2.resize(normal, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    points = cv2.resize(points, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.uint8), (ori_shape[1], ori_shape[0]), cv2.INTER_NEAREST).astype(bool)

    # point cloud
    threshold = 0.04 # 0.01
    mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=threshold)
    if normal is None:
        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
            points,
            image_rgb.astype(np.float32) / 255,
            utils3d.numpy.image_uv(width=ori_shape[1], height=ori_shape[0]),
            mask=mask_cleaned,
            tri=True
        )
        vertex_normals = None
    else:
        faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
            points,
            image_rgb.astype(np.float32) / 255,
            utils3d.numpy.image_uv(width=ori_shape[1], height=ori_shape[0]),
            normal,
            mask=mask_cleaned,
            tri=True
        )
    # When exporting the model, follow the OpenGL coordinate conventions:
    # - world coordinate system: x right, y up, z backward.
    # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
    vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
    if normal is not None:
        vertex_normals = vertex_normals * [1, -1, -1]

    trimesh.Trimesh(
        vertices=vertices, 
        faces=np.zeros((0, 3), dtype=np.int32), 
        vertex_colors=vertex_colors,
        vertex_normals=vertex_normals,
        process=False
    ).export(f'{save_prefix}_point_cloud.ply')

    trimesh.Trimesh(
        vertices=vertices, 
        vertex_normals=vertex_normals,
        faces=faces, 
        visual = trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs, 
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(image_rgb),
                metallicFactor=0.5,
                roughnessFactor=1.0
            )
        ),
        process=False
    ).export(f'{save_prefix}_mesh.glb')

    infer_performace(model)


if __name__ == '__main__':
    main()
