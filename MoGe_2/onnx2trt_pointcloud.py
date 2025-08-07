# by yhpark 2025-7-28
# MoGe-2 TensorRT model generation
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import tensorrt as trt
import torch
import matplotlib
matplotlib.use('Agg') # non-GUI mode
import cv2
import numpy as np
import time
import utils3d
import open3d as o3d
import viser

import common
from common import *

from MoGe.moge.utils.geometry_torch import recover_focal_shift

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO

vis_flag = 'viser' # 'viser' or 'open3d'

# ----- start viser server -----
if vis_flag == 'viser':
    server = viser.ViserServer()
    scene = server.scene

def update_point_cloud(points: np.ndarray, colors: np.ndarray = None):
    if colors is None:
        colors = np.ones_like(points)
    scene.add_point_cloud(
        "/dynamic_point_cloud",
        points=points,
        colors=colors,
        point_size=0.01  # 작게 설정 (기본값은 0.05)
    )

def get_engine(onnx_file_path, engine_file_path="", precision='fp32', dynamic_input_shapes=None):
    """Load or build a TensorRT engine based on the ONNX model."""
    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(0) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser, \
                trt.Runtime(TRT_LOGGER) as runtime:
            
            if not os.path.exists(onnx_file_path):
                raise FileNotFoundError(f"[MDET] ONNX file {onnx_file_path} not found.")
            
            parser.parse_from_file(onnx_file_path)
            
            timing_cache = f"{os.path.dirname(engine_file_path)}/{os.path.splitext(os.path.basename(engine_file_path))[0]}_timing.cache"
            common.setup_timing_cache(config, timing_cache)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(2))
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print(f'[MDET] set fp16 model')

            for i_idx in range(network.num_inputs):
                print(f'[MDET] input({i_idx}) name: {network.get_input(i_idx).name}, shape= {network.get_input(i_idx).shape}')
                
            for o_idx in range(network.num_outputs):
                print(f'[MDET] output({o_idx}) name: {network.get_output(o_idx).name}, shape= {network.get_output(o_idx).shape}')
    
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError("[MDET] Failed to build TensorRT engine. Likely due to shape mismatch or model incompatibility.")
            
            engine = runtime.deserialize_cuda_engine(plan)
            common.save_timing_cache(config, timing_cache)

            with open(engine_file_path, "wb") as f:
                f.write(plan)
            
            return engine

    print(f"[MDET] Engine file path: {engine_file_path}")

    if os.path.exists(engine_file_path):
        print(f"[MDET] Reading engine from file {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        begin = time.time()
        engine = build_engine()
        build_time = time.time() - begin
        build_time_str = f"{build_time:.2f} [sec]" if build_time < 60 else f"{build_time // 60 :.1f} [min] {build_time % 60 :.2f} [sec]"
        print(f'[MDET] engine build done! ({build_time_str})')

        return engine

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

    input_h, input_w = 291, 518

    # Input
    image_dir_name = 'video_frames'
    image_dir = os.path.join(CUR_DIR, '..',image_dir_name)
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]

    # List all files in the directory and filter only image files
    image_paths = [
        os.path.join(image_dir, fname) for fname in sorted(os.listdir(image_dir)) if os.path.splitext(fname)[1].lower() in valid_exts
    ]

    raw_image = cv2.imread(image_paths[0])
    ori_shape = raw_image.shape[:2]
    print(f"[MDET] original image size : {raw_image.shape[:2]}")

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
    
    # ----- init open3d visualization -----
    if vis_flag == 'open3d':
        global_pcd = o3d.geometry.PointCloud()
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Real-time Point Cloud", width=960, height=540)
    
    dur_time = 0
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
        
        output_shape = {}
        for i in range(engine.num_io_tensors):
            output_shape[engine.get_tensor_name(i)] = engine.get_tensor_shape(engine.get_tensor_name(i))
            print(f'[MDET] trt output shape ({engine.get_tensor_name(i)}) : {engine.get_tensor_shape(engine.get_tensor_name(i))}')

        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
                
        # Warm-up      
        for _ in range(20):  
            common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        # Inference loop
        print('start')
        for idx, path in enumerate(image_paths):
            raw_image = cv2.imread(path)
            image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            image_rgb_resized = cv2.resize(image_rgb, (input_w, input_h))
            input_image = preprocess_image(image_rgb_resized)  # Preprocess image
            batch_images = np.concatenate([input_image], axis=0)

            begin = time.time()
            inputs[0].host = batch_images
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

            # # Reshape output
            points = torch.from_numpy(trt_outputs[0].reshape(output_shape['points']))
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
            points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)

            # Apply metric scale
            if metric_scale is not None:
                if points is not None:
                    points *= metric_scale[:, None, None, None]

            # Apply mask
            if mask_binary is not None:
                points = torch.where(mask_binary[..., None], points, torch.inf) if points is not None else None
        
            points = points.numpy().squeeze(0)
            intrinsics = intrinsics.numpy().squeeze(0)

            # original size
            # points = cv2.resize(points, (ori_shape[1], ori_shape[0]), cv2.INTER_LINEAR)
            # colors = np.array(image_rgb).reshape(-1, 3) / 255.0

            colors = np.array(image_rgb_resized).reshape(-1, 3) / 255.0

            # filter
            points = points.reshape(-1, 3)
            mask = np.isfinite(points).all(axis=1)
            filtered_points = points[mask]
            filtered_colors = colors[mask]
            print(f'count : {idx}')

            # ----- update point cloud (viser) -----
            if vis_flag == 'viser':
                update_point_cloud(filtered_points, filtered_colors)
                time.sleep(0.1)

            # ----- update point cloud (open3d) -----
            if vis_flag == 'open3d':
                local_pcd = o3d.geometry.PointCloud()
                local_pcd.points = o3d.utility.Vector3dVector(filtered_points)
                local_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

                voxel_size = 0.01
                local_pcd = local_pcd.voxel_down_sample(voxel_size)
                if len(filtered_points) > 0:
                    if idx == 0:
                        global_pcd = local_pcd
                        vis.add_geometry(global_pcd)

                        view_ctl = vis.get_view_control()
                        bbox = global_pcd.get_axis_aligned_bounding_box()
                        view_ctl.set_lookat(bbox.get_center())
                        view_ctl.set_zoom(0.4)
                        view_ctl.set_front([0.0, 0.0, -1.0])
                        view_ctl.set_up([0.0, -1.0, 0.0])
                    else:
                        if 0:
                            # 누적 전 노말 계산 (global_pcd 기준)
                            global_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                            
                            # ICP 수행
                            reg = o3d.pipelines.registration.registration_icp(
                                source=local_pcd,
                                target=global_pcd,
                                max_correspondence_distance=0.05,
                                init=np.eye(4),
                                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

                            local_pcd.transform(reg.transformation)

                        global_points = np.vstack((np.asarray(global_pcd.points), np.asarray(local_pcd.points)))
                        global_colors = np.vstack((np.asarray(global_pcd.colors), np.asarray(local_pcd.colors)))
                        
                        # points/colors update
                        global_pcd.points = o3d.utility.Vector3dVector(global_points)
                        global_pcd.colors = o3d.utility.Vector3dVector(global_colors)

                        vis.update_geometry(global_pcd)
                        vis.poll_events()
                        vis.update_renderer()
                        time.sleep(0.01)
                        
                else:
                    pass
        
        # ----- free resource (open3d) -----
        if vis_flag == 'open3d':
            vis.run()
            vis.destroy_window()

        # Results
        iteration = len(image_paths)
        print(f'[MDET] {iteration} iterations time ({input_h, input_w}): {dur_time:.4f} [sec]')
        avg_time = dur_time / iteration
        print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
        print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

    common.free_buffers(inputs, outputs, stream)


if __name__ == '__main__':
    main()
