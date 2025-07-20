# by yhpark 2025-7-20
# Distill Any Depth TensorRT model generation
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import tensorrt as trt
import torch
import torch.nn.functional as F

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib

import cv2
import numpy as np
import time
import common
from common import *

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
            if dynamic_input_shapes is not None :
                profile = builder.create_optimization_profile()
                for i_idx in range(network.num_inputs):
                    input = network.get_input(i_idx)
                    assert input.shape[0] == -1
                    min_shape = dynamic_input_shapes[0]
                    opt_shape = dynamic_input_shapes[1]
                    max_shape = dynamic_input_shapes[2]
                    profile.set_shape(input.name, min_shape, opt_shape, max_shape) # any dynamic input tensors
                    print("[TRT_E] Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(input.name, min_shape, opt_shape, max_shape))
                config.add_optimization_profile(profile)

            for i_idx in range(network.num_inputs):
                print(f'[TRT_E] input({i_idx}) name: {network.get_input(i_idx).name}, shape= {network.get_input(i_idx).shape}')
                
            for o_idx in range(network.num_outputs):
                print(f'[TRT_E] output({o_idx}) name: {network.get_output(o_idx).name}, shape= {network.get_output(o_idx).shape}')
    
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


def constrain_to_multiple_of(x, min_val=0, max_val=None, ensure_multiple_of=14):
    y = (np.round(x / ensure_multiple_of) * ensure_multiple_of).astype(int)

    if max_val is not None and y > max_val:
        y = (np.floor(x / ensure_multiple_of) * ensure_multiple_of).astype(int)

    if y < min_val:
        y = (np.ceil(x / ensure_multiple_of) * ensure_multiple_of).astype(int)

    return y

def preprocess_image(raw_image, input_size=518, precision=torch.float32):

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    width, height = image.shape[1], image.shape[0]
    scale_height = input_size / height
    scale_width = input_size / width 

    # scale such that output size is lower bound
    if scale_width > scale_height:
        # fit width
        scale_height = scale_width
    else:
        # fit height
        scale_width = scale_height

    new_height = constrain_to_multiple_of(scale_height * height, min_val=input_size)
    new_width = constrain_to_multiple_of(scale_width * width, min_val=input_size)

    # resize sample
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # NormalizeImage
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # PrepareForNet
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
    print(f'h : {h}, w : {w}')
    print(f'original shape : {raw_img.shape}')
    raw_img = cv2.resize(raw_img, (518, 518))

    input_image = preprocess_image(raw_img)  # Preprocess image
    print(f'after preprocess shape : {input_image.shape}')
    batch_images = np.concatenate([input_image], axis=0)

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    encoder = 'small' # 'large' or 'base', 'small'
    model_name = f"distill_any_depth_{encoder}"
    if 1:
        onnx_model_path = os.path.join(cur_dir, 'onnx', f'{model_name}_sim.onnx')
        engine_file_path = os.path.join(cur_dir, 'engine', f'{model_name}_{precision}_sim.engine')
    else :
        onnx_model_path = os.path.join(cur_dir, 'onnx', f'{model_name}.onnx')
        engine_file_path = os.path.join(cur_dir, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes 
    input_shape = (batch_images.shape)
    output_shape = (1, batch_images.shape[2], batch_images.shape[3])
    print(f'trt input shape : {input_shape}')
    print(f'trt output shape : {output_shape}')
    #dynamic_input_shapes = [[1,3,259,259], [1,3,518,518], [1,3,686,686]]
    #dynamic_input_shapes = [[1,3,518,518], [1,3,518,518], [1,3,518,518]]
    dynamic_input_shapes = None

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision, dynamic_input_shapes) as engine, \
            engine.create_execution_context() as context:
                
        # inspector = engine.create_engine_inspector()
        # inspector.execution_context = context # OPTIONAL
        # print(inspector.get_layer_information(0, trt.tensorrt.LayerInformationFormat.JSON)) # Print the information of the first layer in the engine.
        # print(inspector.get_engine_information( trt.tensorrt.LayerInformationFormat.JSON)) # Print the information of the entire engine.
        
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
        inputs[0].host = batch_images
        
        context.set_input_shape('input', batch_images.shape)
        
        # Warm-up      
        for _ in range(10):  
            common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        # Inference loop
        for _ in range(iteration):
            begin = time.time()
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    # Results
    print(f'[TRT] {iteration} iterations time: {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[TRT] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT] Average inference time: {avg_time * 1000:.2f} [msec]')

    # # Reshape output
    output = torch.from_numpy(trt_outputs[0].reshape(output_shape))
    output = F.interpolate(output[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
    depth0 = output.numpy() 

    # post process
    print(f'max : {depth0.max()} , min : {depth0.min()}')

    depth = (depth0 - depth0.min()) / (depth0.max() - depth0.min()) * 255.0
    depth = depth.astype(np.uint8)
    #cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    cmap = matplotlib.colormaps.get_cmap('turbo')
    heat_map = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    outdir = 'results'
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(cur_dir, outdir, file_name + f'_{encoder}_dad_TRT.png'), heat_map)

    if 1 : # prev version 
        #activation_map = (inverse_depth - np.min(inverse_depth)) / np.max(inverse_depth)
        heat_map = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO) # hw -> hwc
        # 샘플 결과 출력 및 저장
        save_path = os.path.join(cur_dir, outdir, f'{file_name}_dad_TRT_cv.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, heat_map)
    else : # original ml-pro 
        # Save as color-mapped "turbo" jpg image.
        cmap = plt.get_cmap("turbo")
        heat_map = (cmap(depth)[..., :3] * 255).astype(np.uint8)
        # 샘플 결과 출력 및 저장
        save_path = os.path.join(cur_dir, outdir, f'{file_name}_dad_TRT.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(heat_map).save(save_path, format="JPEG", quality=90)

    output_file_npz = os.path.join(cur_dir, outdir, f'{file_name}_dad_TRT')
    np.savez_compressed(output_file_npz, depth=depth)

    common.free_buffers(inputs, outputs, stream)


if __name__ == '__main__':
    main()
