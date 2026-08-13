# by yhpark 2026-3-5
# Metric Anything TensorRT model generation
import os
import sys

# Repo root, found by walking up to the directory holding core/ rather
# than counting "..". Counting breaks the moment a script moves, and
# this one has moved: Phase 4 put every model under models/.
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "core")) and os.path.dirname(_R) != _R:
    _R = os.path.dirname(_R)
sys.path.insert(1, _R)

import tensorrt as trt
import torch
import matplotlib

matplotlib.use("Agg")  # non-GUI mode
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
import utils3d
import trimesh
import trimesh.visual
from PIL import Image

import common
from common import *
from core import bench

import json

sys.path.insert(1, os.path.join(sys.path[0], "metric_anything/models/student_pointmap"))
from moge.utils.geometry_torch import recover_focal_shift

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO




def preprocess_image(raw_image):
    image = raw_image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image).astype(np.float32)
    # [C, H, W] -> [1, C, H, W]
    image = np.expand_dims(image, axis=0)
    return image


# resize_image() with its three modes lived here, but a static engine can only
# use one size, so main() now pins it. infer.py keeps the full version for the
# PyTorch path, where the size can follow the image.


def main():
    save_dir_path = os.path.join(CUR_DIR, "results")
    os.makedirs(save_dir_path, exist_ok=True)

    # Input
    image_file_name = "example.jpg"
    image_path = os.path.join(_R, "data", image_file_name)

    raw_image = cv2.imread(image_path)
    ori_shape = raw_image.shape[:2]
    print(f"[MDET] original image size : {ori_shape}")
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    # 388x518, not 518x518. The square stretch fed aspect_ratio = 1.0 into the
    # intrinsics below and produced a square field of view for a 4:3 photo.
    # Measured on data/example.jpg:
    #
    #   input               fov_x    fov_y   fx      fy
    #   original 3024x2268  58.37    45.46   0.8952  1.1936   <- 4:3, correct
    #   388x518             53.37    41.26   0.9947  1.3280   <- 4:3, correct
    #   518x518 (was)       49.62    49.62   1.0816  1.0816   <- square, wrong
    #
    # The depth map survived either way (7.1% vs 12.1% AbsRel, correlation
    # 0.996+ both), so the damage was confined to the intrinsics and therefore
    # to the point cloud that demo_pointcloud.py builds from them.
    #
    # MoGe_2 runs this identical post-process and already used 388x518, so the
    # two are now consistent. Pinned rather than derived from the source with
    # resize_image(mode=2): the engine is static and its filename carries the
    # size, so deriving it would send a 16:9 photo looking for a 291x518
    # engine that was never built. That does mean 388x518 assumes a 4:3
    # source, the same assumption MoGe_2 makes. See docs/model_contracts.md D13.
    input_h, input_w = 388, 518
    new_height, new_width = input_h, input_w
    re_image_rgb = cv2.resize(image_rgb, (input_w, input_h),
                              interpolation=cv2.INTER_AREA)

    input_image = preprocess_image(re_image_rgb)  # Preprocess image
    print(f"[MDET] after preprocess shape : {input_image.shape}")
    batch_images = np.concatenate([input_image], axis=0)

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    dynamo = False  # True or False
    onnx_sim = False  # True or False
    model_name = f"metric_anything_{new_height}x{new_width}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    onnx_model_path = os.path.join(CUR_DIR, "onnx", f"{model_name}.onnx")
    engine_file_path = os.path.join(
        CUR_DIR, "engine", f"{model_name}_{precision}.engine"
    )
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes
    input_shape = batch_images.shape
    print(f"[MDET] trt input shape : {input_shape}")

    iteration = 100
    warmup = 20
    # Load or build the TensorRT engine and do inference
    with get_engine(
        onnx_model_path, engine_file_path, precision
    ) as engine, engine.create_execution_context() as context:

        output_shape = {}
        for i in range(engine.num_io_tensors):
            output_shape[engine.get_tensor_name(i)] = engine.get_tensor_shape(
                engine.get_tensor_name(i)
            )
            print(
                f"[MDET] trt output shape ({engine.get_tensor_name(i)}) : {engine.get_tensor_shape(engine.get_tensor_name(i))}"
            )

        inputs, outputs, bindings, stream = common.allocate_buffers(
            engine, output_shape, profile_idx=0
        )
        inputs[0].host = batch_images

        # Warm-up and timed loop both live in core/bench.py so every model
        # measures the same thing. warmup is 20 everywhere now; it used to vary
        # between 5 and 20, which alone made two models' numbers incomparable.
        trt_outputs, samples = bench.measure(
            lambda: common.do_inference(context, engine=engine, bindings=bindings,
                                        inputs=inputs, outputs=outputs, stream=stream),
            warmup=warmup, iterations=iteration)

        # Results - printed as before, and written to reports/bench/ so
        # compare.py can build the table without anyone retyping a number.
        bench.record('metric_anything', samples, encoder='fixed', warmup=warmup, precision=precision,
                     profile='bench', input_h=new_height, input_w=new_width,
                     engine_path=engine_file_path,
                     model_input=batch_images)

        # # Reshape output
        points = torch.from_numpy(trt_outputs[0].reshape(output_shape["points"]))
        mask = torch.from_numpy(trt_outputs[1].reshape(output_shape["mask"]))
        metric_scale = torch.from_numpy(
            trt_outputs[2].reshape(output_shape["metric_scale"])
        )
        if mask is not None:
            mask_binary = mask > 0.5
        else:
            mask_binary = None

        focal, shift = recover_focal_shift(points, mask_binary)
        aspect_ratio = input_image.shape[3] / input_image.shape[2]
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

    points = points.numpy().squeeze(0)
    depth = depth.numpy().squeeze(0)
    mask = mask_binary.numpy().squeeze(0)
    intrinsics = intrinsics.numpy().squeeze(0)

    # visualization
    save_prefix = os.path.join(
        save_dir_path, f"{os.path.splitext(image_file_name)[0]}_ma_trt"
    )
    # fov
    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
    with open(f"{save_prefix}_fov.json", "w") as f:
        json.dump(
            {
                "fov_x": round(float(np.rad2deg(fov_x).item()), 2),
                "fov_y": round(float(np.rad2deg(fov_y).item()), 2),
            },
            f,
        )

    # depth
    def clamp(x, min_val, max_val):
        return max(min_val, min(x, max_val))

    depth_ = depth.copy()
    finite_arr = depth_[np.isfinite(depth_)]
    max_val = np.max(finite_arr)
    depth = np.clip(depth, 1e-3, max_val)
    print(f"[MDET] max : {depth.max():.3f} , min : {depth.min():.3f}")

    inverse_depth = 1 / depth
    max_invdepth_vizu = np.nanquantile(inverse_depth, 0.99)
    min_invdepth_vizu = np.nanquantile(inverse_depth, 0.001)
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )

    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)
    color_depth_bgr = cv2.resize(
        color_depth_bgr, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR
    )
    cv2.imwrite(f"{save_prefix}_depth.jpg", color_depth_bgr)

    # save colored depth image with color depth bar
    plt.figure(figsize=(8, 6))
    inverse_depth_normalized_resized = cv2.resize(
        inverse_depth_normalized, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR
    )
    img = plt.imshow(inverse_depth_normalized_resized, cmap="turbo")
    plt.axis("off")
    cbar = plt.colorbar(img, fraction=0.046, pad=0.04)

    num_ticks = 5
    cbar_ticks = np.linspace(0, 1, num_ticks)
    cbar_ticklabels = np.linspace(depth.max(), depth.min(), num_ticks)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{v:.2f} m" for v in cbar_ticklabels])
    cbar.set_label("Depth (m)", fontsize=12)

    plt.tight_layout()
    plt.savefig(
        f"{save_prefix}_depth_bar.jpg", bbox_inches="tight", pad_inches=0.1, dpi=300
    )
    plt.close()

    depth = cv2.resize(depth, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    points = cv2.resize(points, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    mask = cv2.resize(
        mask.astype(np.uint8), (ori_shape[1], ori_shape[0]), cv2.INTER_NEAREST
    ).astype(bool)

    threshold = 0.04  # 0.01
    mask_cleaned = mask & ~utils3d.numpy.depth_map_edge(depth, rtol=threshold)
    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.build_mesh_from_map(
        points,
        image_rgb.astype(np.float32) / 255,
        utils3d.numpy.uv_map(ori_shape[0], ori_shape[1]),
        mask=mask_cleaned,
        tri=True,
    )
    vertex_normals = None

    # When exporting the model, follow the OpenGL coordinate conventions:
    # - world coordinate system: x right, y up, z backward.
    # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
    vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

    trimesh.Trimesh(
        vertices=vertices,
        faces=np.zeros((0, 3), dtype=np.int32),
        vertex_colors=vertex_colors,
        process=False,
    ).export(f"{save_prefix}_point_cloud.ply")

    trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(image_rgb),
                metallicFactor=0.5,
                roughnessFactor=1.0,
            ),
        ),
        process=False,
    ).export(f"{save_prefix}_mesh.glb")

    common.free_buffers(inputs, outputs, stream)


if __name__ == "__main__":
    main()
