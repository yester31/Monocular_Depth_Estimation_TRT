# by yhpark 2025-7-28
# MoGe-2 TensorRT model generation
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
matplotlib.use('Agg') # non-GUI mode
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
from MoGe.moge.utils.geometry_torch import recover_focal_shift

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

def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    input_h, input_w = 388, 518 # 291, 518 # 700, 700

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(_R, 'data', image_file_name)
    # image_file_name = '7.jpg'
    # image_path = os.path.join(_R, 'StreamVGGT','StreamVGGT','examples','example_building', image_file_name)
    # image_file_name = 'frame_00000.png'
    # image_path = os.path.join(_R, 'video_frames_50', image_file_name)
    raw_image = cv2.imread(image_path)
    ori_shape = raw_image.shape[:2]
    print(f"[MDET] original image size : {ori_shape}")
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    image_rgb_resized = cv2.resize(image_rgb, (input_w, input_h))

    input_image = preprocess_image(image_rgb_resized)  # Preprocess image
    print(f'[MDET] after preprocess shape : {input_image.shape}')
    batch_images = np.concatenate([input_image], axis=0)

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    encoder = 'vits' # 'vitl' or 'vitb', 'vits'
    dynamo = True      # True or False
    onnx_sim = True    # True or False
    model_name = f"moge-2_{encoder}_normal_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes 
    input_shape = (batch_images.shape)
    print(f'[MDET] trt input shape : {input_shape}')

    iteration = 100
    warmup = 20
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
        
        output_shape = {}
        for i in range(engine.num_io_tensors):
            output_shape[engine.get_tensor_name(i)] = engine.get_tensor_shape(engine.get_tensor_name(i))
            print(f'[MDET] trt output shape ({engine.get_tensor_name(i)}) : {engine.get_tensor_shape(engine.get_tensor_name(i))}')

        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
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
        bench.record('moge_2', samples, warmup=warmup, precision=precision,
                     profile='bench', input_h=input_h, input_w=input_w,
                     engine_path=engine_file_path,
                     model_input=batch_images)

        # # Reshape output
        points = torch.from_numpy(trt_outputs[0].reshape(output_shape['points']))
        normal = torch.from_numpy(trt_outputs[1].reshape(output_shape['normal']))
        mask = torch.from_numpy(trt_outputs[2].reshape(output_shape['mask']))
        metric_scale = torch.from_numpy(trt_outputs[3].reshape(output_shape['metric_scale']))
        if mask is not None:
            mask_binary = mask > 0.5
        else:
            mask_binary = None

        focal, shift = recover_focal_shift(points, mask_binary)
        aspect_ratio = input_image.shape[3] / input_image.shape[2]
        fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
        points[..., 2] += shift[..., None, None]
        if mask_binary is not None:
            mask_binary &= points[..., 2] > 0        # in case depth is contains negative values (which should never happen in practice)
        depth = points[..., 2].clone()

        # depth_map_to_point_map, not depth_to_points. The latter does not
        # exist in the utils3d commit both MoGe and Metric Anything pin
        # (3fab839f); Metric_Anything/onnx2trt.py already calls the right
        # one. Same signature, so this is a rename.
        points = utils3d.torch.depth_map_to_point_map(depth, intrinsics=intrinsics)

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
    
    points = points.numpy().squeeze(0)
    depth = depth.numpy().squeeze(0)
    normal = normal.numpy().squeeze(0)
    mask = mask_binary.numpy().squeeze(0)
    intrinsics = intrinsics.numpy().squeeze(0)

    # visualization
    save_prefix = os.path.join(save_dir_path, f'{os.path.splitext(image_file_name)[0]}_{encoder}_m2_trt')
    # fov
    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
    with open(f'{save_prefix}_fov.json', 'w') as f:
        json.dump({'fov_x': round(float(np.rad2deg(fov_x).item()), 2), 'fov_y': round(float(np.rad2deg(fov_y).item()), 2),}, f)
    
    # normal
    normal2 = normal * [0.5, -0.5, -0.5] + 0.5
    normal2 = (normal2.clip(0, 1) * 255).astype(np.uint8)
    color_normal_bgr = cv2.cvtColor(normal2, cv2.COLOR_RGB2BGR)
    color_normal_bgr = cv2.resize(color_normal_bgr, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    cv2.imwrite(f'{save_prefix}_normal.jpg', color_normal_bgr)

    # depth
    def clamp(x, min_val, max_val):
        return max(min_val, min(x, max_val))

    print(f'[MDET] max : {depth.max()} , min : {depth.min()}')
    depth_ = depth.copy() 
    finite_arr = depth_[np.isfinite(depth_)]
    max_val = np.max(finite_arr)
    depth = np.clip(depth, 1e-3, max_val)
    print(f'[MDET] max : {depth.max()} , min : {depth.min()}')

    inverse_depth = 1 / depth
    max_invdepth_vizu = np.nanquantile(inverse_depth, 0.99)
    min_invdepth_vizu = np.nanquantile(inverse_depth, 0.001)
    #max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1) # from dpeth pro
    #min_invdepth_vizu = max(1 / 250, inverse_depth.min()) # from dpeth pro
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)

    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    color_depth_bgr = cv2.resize(color_depth_bgr, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    cv2.imwrite(f'{save_prefix}_depth.jpg', color_depth_bgr)

    # save colored depth image with color depth bar
    plt.figure(figsize=(8, 6))
    inverse_depth_normalized_resized = cv2.resize(inverse_depth_normalized, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    img = plt.imshow(inverse_depth_normalized_resized, cmap='turbo')  
    plt.axis('off')
    cbar = plt.colorbar(img, fraction=0.046, pad=0.04)

    num_ticks = 5
    cbar_ticks = np.linspace(0, 1, num_ticks)
    cbar_ticklabels = np.linspace(depth.max(), depth.min(),  num_ticks)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{v:.2f} m' for v in cbar_ticklabels])
    cbar.set_label('Depth (m)', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_depth_bar.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

    # Everything below writes a textured mesh, and it is off by default because
    # it does not run against the utils3d commit MoGe pins (3fab839f). Three
    # names it calls were renamed upstream, with different signatures:
    #
    #   depth_edge  -> depth_map_edge
    #   image_uv    -> uv_map
    #   image_mesh  -> build_mesh_from_map     (returns a different tuple)
    #
    # So this was written against some other utils3d than the one MoGe asks
    # for. Porting it means re-deriving the mesh construction, which cannot be
    # checked by reading -- it needs the output looked at. Left intact and
    # disabled rather than half-translated, since a mesh that builds but is
    # subtly wrong is worse than one that does not build.
    #
    # None of this affects the benchmark: bench.record() has already written
    # the result by this point. MoGe_2/onnx2trt_pointcloud.py is the dedicated
    # geometry script and carries the same calls, so it needs the same port.
    export_mesh = False
    if not export_mesh:
        return

    depth = cv2.resize(depth, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    normal = cv2.resize(normal, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    points = cv2.resize(points, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.uint8), (ori_shape[1], ori_shape[0]), cv2.INTER_NEAREST).astype(bool)

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

    common.free_buffers(inputs, outputs, stream)


if __name__ == '__main__':
    main()
