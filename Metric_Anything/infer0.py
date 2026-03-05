import argparse
import itertools
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm
import sys

sys.path.insert(1, os.path.join(sys.path[0], "metric_anything/models/student_pointmap"))
from moge.model.v2 import MoGeModel
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, save_plt
import utils3d

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
dtype = torch.half
print(f"[MDET] Using dtype: {dtype}")


def infer_performace(model, input_size=(518, 518)):
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(DEVICE).half()

    with torch.no_grad():
        for _ in range(20):
            pred_disp = model.infer(dummy_input, fov_x=None)
    torch.cuda.synchronize()

    iteration = 100
    dur_time = 0
    with torch.no_grad():
        for _ in range(iteration):
            begin = time.time()
            pred_disp = model.infer(dummy_input, fov_x=None)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    print(f"[MDET] input size : {input_size[0]} x {input_size[1]} ")
    print(f"[MDET] dtype: {dtype}")
    print(f"[MDET] {iteration} iterations time : {dur_time:.4f} [sec]")
    avg_time = dur_time / iteration
    print(f"[MDET] Average FPS: {1 / avg_time:.2f} [fps]")
    print(f"[MDET] Average inference time: {avg_time * 1000:.2f} [msec]")


def main():

    save_dir_path = os.path.join(CUR_DIR, "results")
    os.makedirs(save_dir_path, exist_ok=True)

    print("[MDET] Load model & image")
    checkpoint = f"{CUR_DIR}/checkpoints/student_pointmap.pt"
    model = MoGeModel.from_pretrained(checkpoint).to(DEVICE)
    model.eval()
    if dtype == torch.half:
        model.half()

    # input
    image_file_name = "example.jpg"
    image_path = os.path.join(CUR_DIR, "..", "data", image_file_name)
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (518, 518))
    # ===================================================================
    print("[MDET] Pre process")
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height}, {width}")
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    image = image_rgb / 255.0
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)
    if dtype == torch.half:
        image_tensor = image_tensor.half()
    print(f"[MDET] model input size : {image_tensor.shape}")

    # ===================================================================
    print("[MDET] Run inference")
    # Inference
    with torch.no_grad():
        output = model.infer(image_tensor, fov_x=None)

    # ===================================================================
    print("[MDET] Post process")
    points = output["points"].cpu().numpy().squeeze(0)
    depth = output["depth"].cpu().numpy().squeeze(0)
    mask = output["mask"].cpu().numpy().squeeze(0)
    intrinsics = output["intrinsics"].cpu().numpy().squeeze(0)

    save_prefix = os.path.join(
        save_dir_path,
        f"{os.path.splitext(image_file_name)[0]}_ma_torch_ori_{height}_{width}",
    )

    # depth
    vis_depth = colorize_depth(depth, cmap="turbo_r")
    color_depth_bgr = cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{save_prefix}_depth.jpg", color_depth_bgr)

    # depth with bar
    save_plt(
        save_dir_path,
        depth,
        mask,
        os.path.splitext(image_file_name)[0] + f"_ma_torch_ori_depth_bar.jpg",
    )

    # mask
    cv2.imwrite(f"{save_prefix}_mask.jpg", (mask * 255).astype(np.uint8))

    # fov
    fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
    with open(f"{save_prefix}_fov.json", "w") as f:
        json.dump(
            {
                "fov_x": round(float(np.rad2deg(fov_x)), 2),
                "fov_y": round(float(np.rad2deg(fov_y)), 2),
            },
            f,
            indent=4,
        )

    # point cloud
    mask_cleaned = mask & ~utils3d.np.depth_map_edge(depth, rtol=0.1)
    faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
        points,
        image_rgb.astype(np.float32) / 255.0,
        utils3d.np.uv_map(height, width),
        mask=mask_cleaned,
        tri=True,
    )

    vertices = vertices * np.array([1, -1, -1])
    vertex_uvs = vertex_uvs * np.array([1, -1]) + np.array([0, 1])
    save_glb(f"{save_prefix}_mesh.glb", vertices, faces, vertex_uvs, image_rgb, None)
    save_ply(
        f"{save_prefix}_pointcloud.ply",
        vertices,
        np.zeros((0, 3), dtype=np.int32),
        vertex_colors,
        None,
    )

    # infer_performace(model)
    print(f"[MDET] max : {depth.max():.3f} , min : {depth.min():.3f}")


if __name__ == "__main__":
    main()
