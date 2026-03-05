# by yhpark 2026-3-4
import os
from typing import *
import json
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import time
import sys

sys.path.insert(1, os.path.join(sys.path[0], "metric_anything/models/student_pointmap"))
from moge.model.v2 import MoGeModel
from moge.utils.geometry_torch import recover_focal_shift
import utils3d
import trimesh
import trimesh.visual
from PIL import Image

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")


def infer_performace(model, input_size=(518, 518)):
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(DEVICE).half()
    num_tokens = torch.tensor(3600).to(DEVICE).half()

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

    print(f"[MDET] {iteration} iterations time ({input_size}): {dur_time:.4f} [sec]")
    avg_time = dur_time / iteration
    print(f"[MDET] Average FPS: {1 / avg_time:.2f} [fps]")
    print(f"[MDET] Average inference time: {avg_time * 1000:.2f} [msec]")


def set_model(dtype: torch.dtype = torch.float32):

    checkpoint = f"{CUR_DIR}/checkpoints/student_pointmap.pt"

    model = MoGeModel.from_pretrained(checkpoint).to(DEVICE)
    model = model.eval()

    if dtype == torch.half:
        model.half()

    return model


def resize_image(image_rgb, resize_mode=0, resize_to=518):
    """
    image resize function

    resize_mode
    0 : original size
    1 : resize to 518 x 518
    2 : resize to 518 (keep aspect ratio)
    """

    height, width = image_rgb.shape[:2]

    # 0. original size
    if resize_mode == 0:
        new_height = height
        new_width = width
        print(f"[MDET] original size : {(new_height, new_width)}")

    # 1. resize to 518 x 518
    elif resize_mode == 1:
        new_height = resize_to
        new_width = resize_to
        print(f"[MDET] resize_to 518x518 : {(new_height, new_width)}")

    # 2. resize to 518 (keep aspect ratio)
    elif resize_mode == 2:
        new_height = min(resize_to, int(resize_to * height / width))
        new_width = min(resize_to, int(resize_to * width / height))
        print(f"[MDET] resize_to 518 keep aspect ratio : {(new_height, new_width)}")

    else:
        raise ValueError("resize_mode must be 0, 1, or 2")

    # resize 수행
    resized_image = cv2.resize(image_rgb, (new_width, new_height), cv2.INTER_AREA)

    return resized_image, (new_height, new_width)


def main():

    save_dir_path = os.path.join(CUR_DIR, "results")
    os.makedirs(save_dir_path, exist_ok=True)

    print("[MDET] Load model & image")
    dtype = torch.half  # torch.half or torch.float32
    model = set_model(dtype)

    # input
    image_file_name = "example.jpg"
    image_path = os.path.join(CUR_DIR, "..", "data", image_file_name)
    raw_image = cv2.imread(image_path)
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height}, {width}")
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    # ===================================================================
    print("[MDET] Pre process")

    re_image_rgb, (new_height, new_width) = resize_image(
        image_rgb, resize_mode=1, resize_to=518
    )
    image_tensor = torch.from_numpy(re_image_rgb / 255.0)  # [0, 255] -> [0, 1]
    image_tensor = (
        image_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    )  # HWC -> NCHW, to cuda
    if dtype == torch.half:
        image_tensor = image_tensor.half()
    print(f"[MDET] model input size : {image_tensor.shape}")

    # ===================================================================
    print("[MDET] Run inference")
    # Inference
    with torch.no_grad():
        output = model(image_tensor, 3600)
        # ===================================================================
        print("[MDET] Post process")
        points, normal, mask, metric_scale = (
            output.get(k, None) for k in ["points", "normal", "mask", "metric_scale"]
        )
        points, normal, mask, metric_scale = map(
            lambda x: x.float().cpu() if isinstance(x, torch.Tensor) else x,
            [points, normal, mask, metric_scale],
        )

        if mask is not None:
            mask_binary = mask > 0.5
        else:
            mask_binary = None

        focal, shift = recover_focal_shift(points, mask_binary)

        aspect_ratio = new_width / new_height
        fx, fy = (
            focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio,
            focal / 2 * (1 + aspect_ratio**2) ** 0.5,
        )
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
        points[..., 2] += shift[..., None, None]
        if mask_binary is not None:
            mask_binary &= (
                points[..., 2] > 0
            )  # in case depth is contains negative values (which should never happen in practice)
        depth = points[..., 2].clone()

        points = utils3d.torch.depth_map_to_point_map(depth, intrinsics=intrinsics)

        # Apply metric scale
        if metric_scale is not None:
            if points is not None:
                points *= metric_scale[:, None, None, None]
            if depth is not None:
                depth *= metric_scale[:, None, None]

        # Apply mask
        if mask_binary is not None:
            points = (
                torch.where(mask_binary[..., None], points, torch.inf)
                if points is not None
                else None
            )
            depth = (
                torch.where(mask_binary, depth, torch.inf)
                if depth is not None
                else None
            )
            normal = (
                torch.where(mask_binary[..., None], normal, torch.zeros_like(normal))
                if normal is not None
                else None
            )

    points = (
        points.cpu().numpy().squeeze(0)
    )  # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
    depth = depth.cpu().numpy().squeeze(0)  # depth map
    mask = mask_binary.cpu().numpy().squeeze(0)  # a binary mask for valid pixels.
    intrinsics = intrinsics.cpu().numpy().squeeze(0)  # normalized camera intrinsics

    save_prefix = os.path.join(
        save_dir_path,
        f"{os.path.splitext(image_file_name)[0]}_ma_torch_{new_height}_{new_width}",
    )
    # fov
    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
    with open(f"{save_prefix}_fov.json", "w") as f:
        json.dump(
            {
                "fov_x": round(float(np.rad2deg(fov_x)), 2),
                "fov_y": round(float(np.rad2deg(fov_y)), 2),
            },
            f,
            indent=4,
        )

    # depth
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    inverse_depth = 1 / depth
    max_invdepth_vizu = np.nanquantile(inverse_depth, 0.99)
    min_invdepth_vizu = np.nanquantile(inverse_depth, 0.001)
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )

    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)
    # color_depth_bgr = cv2.resize(color_depth_bgr, (new_width, new_height), cv2.INTER_LINEAR)
    cv2.imwrite(f"{save_prefix}_depth.jpg", color_depth_bgr)

    depth = cv2.resize(depth, (new_width, new_height), cv2.INTER_LINEAR)
    points = cv2.resize(points, (new_width, new_height), cv2.INTER_LINEAR)
    mask = cv2.resize(
        mask.astype(np.uint8), (new_width, new_height), cv2.INTER_NEAREST
    ).astype(bool)

    # point cloud
    threshold = 0.04  # 0.01
    mask_cleaned = mask & ~utils3d.np.depth_map_edge(depth, rtol=threshold)
    faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
        points,
        re_image_rgb.astype(np.float32) / 255.0,
        utils3d.np.uv_map(new_height, new_width),
        mask=mask_cleaned,
        tri=True,
    )

    # When exporting the model, follow the OpenGL coordinate conventions:
    # - world coordinate system: x right, y up, z backward.
    # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
    vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

    trimesh.Trimesh(
        vertices=vertices,
        faces=np.zeros((0, 3), dtype=np.int32),
        vertex_colors=vertex_colors,
        # vertex_normals=vertex_normals,
        process=False,
    ).export(f"{save_prefix}_point_cloud.ply")

    trimesh.Trimesh(
        vertices=vertices,
        # vertex_normals=vertex_normals,
        faces=faces,
        visual=trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(re_image_rgb),
                metallicFactor=0.5,
                roughnessFactor=1.0,
            ),
        ),
        process=False,
    ).export(f"{save_prefix}_mesh.glb")

    infer_performace(model)
    print(f"[MDET] max : {depth.max():.3f} , min : {depth.min():.3f}")


if __name__ == "__main__":
    main()
