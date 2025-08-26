# by yhpark 2025-8-5
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import tensorrt as trt
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt

import cv2
import numpy as np
import time
import common
from common import *

import sys
sys.path.insert(1, os.path.join(sys.path[0], "vggt"))
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri

import open3d as o3d

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

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
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(4))
            # config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print(f'[MDET] set fp16 model')

            for i_idx in range(network.num_inputs):
                print(f'[MDET] input({i_idx}) name: {network.get_input(i_idx).name}, shape= {network.get_input(i_idx).shape}')
                
            for o_idx in range(network.num_outputs):
                print(f'[MDET] output({o_idx}) name: {network.get_output(o_idx).name}, shape= {network.get_output(o_idx).shape}')
            
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError("Failed to build TensorRT engine. Likely due to shape mismatch or model incompatibility.")
            
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

def pre_process(raw_image, target_size, input_h, input_w):
    original_coords = []  # Renamed from position_info to be more descriptive
    height, width = raw_image.shape[:2]
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
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    
    # Convert to tensor
    img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
    img = img.unsqueeze(0)
    batch_images = F.interpolate(img, size=(input_h, input_w), mode="bilinear", align_corners=False)
    batch_images = batch_images.unsqueeze(0)
    return batch_images.cpu().numpy(), x1, y1, x2, y2

def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # Input
    target_size = 1024
    input_h, input_w = 518, 518
    image_path = os.path.join(CUR_DIR, '..', 'data', 'example.jpg')
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]

    raw_image = cv2.imread(image_path)
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height, width}")

    print('[MDET] Pre process')
    batch_images, x1, y1, x2, y2 = pre_process(raw_image, target_size, input_h, input_w)

    # Model and engine paths
    onnx_dtype_fp16 = True
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    model_name = f"vggt_aggregator_{input_h}x{input_w}"
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', model_name, f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    model_name2 = f"vggt_depth_head_{input_h}x{input_w}"
    model_name2 = f"{model_name2}_fp16" if onnx_dtype_fp16 else model_name2    
    onnx_model_path2 = os.path.join(CUR_DIR, 'onnx', model_name2, f'{model_name2}.onnx')
    engine_file_path2 = os.path.join(CUR_DIR, 'engine', f'{model_name2}_{precision}.engine')

    model_name3 = f"vggt_camera_head_{input_h}x{input_w}"
    model_name3 = f"{model_name3}_fp16" if onnx_dtype_fp16 else model_name3    
    onnx_model_path3 = os.path.join(CUR_DIR, 'onnx', model_name3, f'{model_name3}.onnx')
    engine_file_path3 = os.path.join(CUR_DIR, 'engine', f'{model_name3}_{precision}.engine')

    engine = get_engine(onnx_model_path, engine_file_path, precision)
    engine2 = get_engine(onnx_model_path2, engine_file_path2, precision)
    engine3 = get_engine(onnx_model_path3, engine_file_path3, precision)

    context = engine.create_execution_context()
    context2 = engine2.create_execution_context()
    context3 = engine3.create_execution_context()

    depth_shape = (input_h, input_w)
    # aggregated_tokens_list_shape = (24, 1, 1, 1374, 2048)
    pose_enc_shape = (1, 1, 9)
    try:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs2, outputs2, bindings2, stream2 = common.allocate_buffers(engine2)
        inputs3, outputs3, bindings3, stream3 = common.allocate_buffers(engine3)

        for i in range(engine.num_io_tensors):
            context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
        for i in range(engine2.num_io_tensors):
            context2.set_tensor_address(engine2.get_tensor_name(i), bindings2[i])
        for i in range(engine3.num_io_tensors):
            context3.set_tensor_address(engine3.get_tensor_name(i), bindings3[i])

        # Transfer input data to the GPU.
        kind_h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        kind_d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        kind_d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

        # Warm-up      
        for _ in range(20):  
            inputs[0].host = batch_images
            context.execute_async_v3(stream_handle=stream)
            context2.execute_async_v3(stream_handle=stream)
            context3.execute_async_v3(stream_handle=stream)
            cuda_call(cudart.cudaStreamSynchronize(stream)) # Synchronize the stream

        # Inference loop
        iteration = 100
        dur_time = 0
        for _ in range(iteration):
            begin = time.time()

            inputs[0].host = batch_images
            cuda_call(cudart.cudaMemcpyAsync(inputs[0].device, inputs[0].host, inputs[0].nbytes, kind_h2d, stream))
            context.execute_async_v3(stream_handle=stream)

            cuda_call(cudart.cudaMemcpyAsync(inputs2[0].device, outputs[0].device, outputs[0].nbytes, kind_d2d, stream))
            context2.execute_async_v3(stream_handle=stream)
            cuda_call(cudart.cudaMemcpyAsync(outputs2[0].host, outputs2[0].device, outputs2[0].nbytes, kind_d2h, stream))

            cuda_call(cudart.cudaMemcpyAsync(inputs3[0].device, outputs[0].device, outputs[0].nbytes, kind_d2d, stream))
            context3.execute_async_v3(stream_handle=stream)
            cuda_call(cudart.cudaMemcpyAsync(outputs3[0].host, outputs3[0].device, outputs3[0].nbytes, kind_d2h, stream))
            cuda_call(cudart.cudaStreamSynchronize(stream)) # Synchronize the stream

            dur_time += time.time() - begin
        # ===================================================================
        # Results
        print(f'[MDET] {iteration} iterations time: {dur_time:.4f} [sec]')
        avg_time = dur_time / iteration
        print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
        print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

        if 0 :
            # aggregator
            inputs[0].host = batch_images   
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            
            # depth_head
            inputs2[0].host = outputs[0].host
            trt_outputs2 = common.do_inference(context2, engine=engine2, bindings=bindings2, inputs=inputs2, outputs=outputs2, stream=stream2)
            
            # camera_head
            inputs3[0].host = outputs[0].host
            trt_outputs3 = common.do_inference(context3, engine=engine3, bindings=bindings3, inputs=inputs3, outputs=outputs3, stream=stream3)

        print('[MDET] Post process')
        depth_map = outputs2[0].host.reshape(depth_shape) # [518, 518, 1]
        depth_conf = outputs2[1].host.reshape(depth_shape) # [518, 518, 1]
        print(f'[MDET] max : {depth_map.max():0.5f} , min : {depth_map.min():0.5f}')

        depth_map = cv2.resize(depth_map, (target_size, target_size), cv2.INTER_LINEAR)
        depth_map = depth_map[int(y1):int(y2), int(x1):int(x2),...] # remove paddings
        depth_map = depth_map[None, :, :, None]
        
        pose_enc = torch.from_numpy(outputs3[0].host.reshape(pose_enc_shape))
        print(f'[MDET] pose_enc : \n {pose_enc}')
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (input_h, input_w))
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsic.squeeze(0), intrinsic.squeeze(0))

    finally:
        del context
        del engine

    # ===================================================================
    print('[MDET] Generate color depth image')

    # visualization
    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    depth_map = np.squeeze(depth_map)
    inverse_depth = 1 / depth_map
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu + 1e-6)

    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    color_depth_bgr = cv2.resize(color_depth_bgr, (width, height), cv2.INTER_LINEAR)

    # save colored depth image 
    output_file_depth = os.path.join(save_dir_path, f"{image_file_name}_vggt_{input_h}x{input_w}_trt2.jpg")
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, f"{image_file_name}_vggt_{input_h}x{input_w}_trt2")
    np.savez_compressed(output_file_npz, depth=depth_map)

    # save point cloud 
    rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) 
    rgb_resized_img = cv2.resize(rgb_image, (depth_map.shape[1], depth_map.shape[0]), cv2.INTER_LINEAR)
    points = np.stack(world_points, axis=-1).reshape(-1, 3)
    colors = np.array(rgb_resized_img).reshape(-1, 3) / 255.0

    if 0 :
        conf_flat = depth_conf.reshape(-1)
        init_conf_threshold = 50.0
        init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
        init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
        scene_center = np.mean(points, axis=0)
        points_centered = points - scene_center
        points = points_centered[init_conf_mask]

    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(save_dir_path, f"{image_file_name}_vggt_{input_h}x{input_w}_trt2.ply"), pcd)
    # o3d.visualization.draw_geometries([pcd])

    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()
