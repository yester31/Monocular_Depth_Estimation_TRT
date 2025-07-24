# by yhpark 2025-7-24
# MoGe-2 TensorRT model generation
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import tensorrt as trt
import torch
import matplotlib
import cv2
import numpy as np
import time
import common
from common import *

import json
import utils3d
from MoGe.moge.utils.geometry_torch import recover_focal_shift
from MoGe.moge.utils.vis import colorize_depth, colorize_normal

current_file_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(current_file_path)
print(f"current file path: {current_file_path}")
print(f"current directory: {cur_dir}")

# Global Variables
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO

def get_engine(onnx_file_path, engine_file_path="", precision='fp32', dynamic_input_shapes=None):
    """Load or build a TensorRT engine based on the ONNX model."""
    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(0) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser, \
                trt.Runtime(TRT_LOGGER) as runtime:
            
            if not os.path.exists(onnx_file_path):
                raise FileNotFoundError(f"[TRT] ONNX file {onnx_file_path} not found.")

            # print(f"[TRT] Loading and parsing ONNX file: {onnx_file_path}")
            # with open(onnx_file_path, "rb") as model:
            #     if not parser.parse(model.read()):
            #         raise RuntimeError("[TRT] Failed to parse the ONNX file.")
            #     for error in range(parser.num_errors):
            #         print(parser.get_error(error))
                    
            parser.parse_from_file(onnx_file_path)
            
            timing_cache = f"{cur_dir}/timing_{os.path.splitext(os.path.basename(onnx_file_path))[0]}.cache"
            common.setup_timing_cache(config, timing_cache)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(2))
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print(f'[TRT_E] set fp16 model')

            for i_idx in range(network.num_inputs):
                print(f'[TRT_E] input({i_idx}) name: {network.get_input(i_idx).name}, shape= {network.get_input(i_idx).shape}')
                
            for o_idx in range(network.num_outputs):
                print(f'[TRT_E] output({o_idx}) name: {network.get_output(o_idx).name}, shape= {network.get_output(o_idx).shape}')
    
            if dynamic_input_shapes is not None :
                profile = builder.create_optimization_profile()
                for i_idx in range(network.num_inputs):
                    input = network.get_input(i_idx)
                    #assert input.shape[0] == -1
                    if input.name == 'num_tokens':
                        profile.set_shape(input.name, min=(1,), opt=(1,), max=(1,)) # any dynamic input tensors
                        continue
                    min_shape = dynamic_input_shapes[0]
                    opt_shape = dynamic_input_shapes[1]
                    max_shape = dynamic_input_shapes[2]
                    profile.set_shape(input.name, min_shape, opt_shape, max_shape) # any dynamic input tensors
                    
                    print("[TRT_E] Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(input.name, min_shape, opt_shape, max_shape))
                config.add_optimization_profile(profile)

            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError("Failed to build TensorRT engine. Likely due to shape mismatch or model incompatibility.")
            
            engine = runtime.deserialize_cuda_engine(plan)
            common.save_timing_cache(config, timing_cache)

            with open(engine_file_path, "wb") as f:
                f.write(plan)
            
            return engine

    print(f"[TRT] Engine file path: {engine_file_path}")

    if os.path.exists(engine_file_path):
        print(f"[TRT] Reading engine from file {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        begin = time.time()
        engine = build_engine()
        build_time = time.time() - begin
        build_time_str = f"{build_time:.2f} [sec]" if build_time < 60 else f"{build_time // 60 :.1f} [min] {build_time % 60 :.2f} [sec]"
        print(f'[TRT_E] engine build done! ({build_time_str})')

        return engine


def preprocess_image(raw_image):
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image).astype(np.float32)
    # [C, H, W] -> [1, C, H, W]
    image = np.expand_dims(image, axis=0)

    return image


def main():
    iteration = 100
    dur_time = 0

    # Input
    image_path = os.path.join(cur_dir, '..', 'data', 'example.jpg')
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    raw_img = cv2.imread(image_path)
    h, w = raw_img.shape[:2]
    print(f'original shape : {raw_img.shape}')
    img_size_h, img_size_w = 518, 518
    raw_img = cv2.resize(raw_img, (img_size_w, img_size_h))

    input_image = preprocess_image(raw_img)  # Preprocess image
    print(f'after preprocess shape : {input_image.shape}')
    batch_images = np.concatenate([input_image], axis=0)

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    encoder = 'vits' # 'vitl' or 'vitb', 'vits'
    model_name = f"moge-2-{encoder}-normal" 

    sim_mode = '_sim' # '_sim' or ''
    shape_mode = 'static' # 'dynamic' or 'static'
    onnx_model_path = os.path.join(cur_dir, 'onnx', f'{model_name}_{shape_mode}{sim_mode}.onnx')
    engine_file_path = os.path.join(cur_dir, 'engine', f'{model_name}_{precision}_{shape_mode}{sim_mode}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes 
    input_shape = (batch_images.shape)
    print(f'trt input shape : {input_shape}')
    if shape_mode == 'dynamic':
        dynamic_input_shapes = [[1,3,256,256], [1,3,518,518], [1,3,1024,1024]]
        # dynamic_input_shapes = [[1,3,1024,1024], [1,3,2048,2048], [1,3,3072,3072]]
        output_shape = {
            "points" : [1,img_size_h,img_size_w,3], 
            "normal" : [1,img_size_h,img_size_w,3], 
            "mask" : [1,img_size_h,img_size_w], 
            "metric_scale" : [1], 
                        }
    else:
        dynamic_input_shapes = None
        output_shape = None

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision, dynamic_input_shapes) as engine, \
            engine.create_execution_context() as context:
        
        if shape_mode == 'static':
            output_shape = {}
            for i in range(engine.num_io_tensors):
                output_shape[engine.get_tensor_name(i)] = engine.get_tensor_shape(engine.get_tensor_name(i))
                print(f'trt output shape ({engine.get_tensor_name(i)}) : {engine.get_tensor_shape(engine.get_tensor_name(i))}')

        # inspector = engine.create_engine_inspector()
        # inspector.execution_context = context # OPTIONAL
        # print(inspector.get_layer_information(0, trt.tensorrt.LayerInformationFormat.JSON)) # Print the information of the first layer in the engine.
        # print(inspector.get_engine_information( trt.tensorrt.LayerInformationFormat.JSON)) # Print the information of the entire engine.
        
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
        inputs[0].host = batch_images
        
        context.set_input_shape('image', batch_images.shape)
        
        # Warm-up      
        for _ in range(20):  
            common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        # Inference loop
        for _ in range(iteration):
            begin = time.time()
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    # Results
    print(f'[TRT] {iteration} iterations time ({raw_img.shape[:2]}): {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[TRT] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT] Average inference time: {avg_time * 1000:.2f} [msec]')

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
    
    points = points.numpy().squeeze(0)
    intrinsics = intrinsics.numpy().squeeze(0)
    depth = depth.numpy().squeeze(0)
    mask = mask_binary.numpy().squeeze(0)
    normal = normal.numpy().squeeze(0)
     

    # visualization
    print(f'max : {depth.max()} , min : {depth.min()}')
    depth = 1 / depth
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    cmap = matplotlib.colormaps.get_cmap('turbo')
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    depth = cv2.resize(depth, (w, h))

    outdir = 'results'
    save_path = os.path.join(cur_dir, outdir)
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, f'{file_name}_{encoder}_moge2_TRT_{shape_mode}{sim_mode}_depth.png'), depth)

    normal = cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_path, f'{file_name}_{encoder}_moge2_TRT_{shape_mode}{sim_mode}_normal.png'), normal)

    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
    with open(f'{save_path}/{file_name}_{encoder}_moge2_TRT_{shape_mode}{sim_mode}_fov.json', 'w') as f:
        json.dump({'fov_x': round(float(np.rad2deg(fov_x)), 2), 'fov_y': round(float(np.rad2deg(fov_y)), 2),}, f)

    common.free_buffers(inputs, outputs, stream)


if __name__ == '__main__':
    main()
