# by yhpark 2025-7-27
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

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
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

            parser.parse_from_file(onnx_file_path)
            
            timing_cache = f"{os.path.dirname(engine_file_path)}/{os.path.splitext(os.path.basename(engine_file_path))[0]}_timing.cache"
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

def preprocess_image(raw_image, input_h=518, input_w=518, precision=torch.float32):

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    width, height = image.shape[1], image.shape[0]
    scale_height = input_h / height
    scale_width = input_w / width 

    new_height = constrain_to_multiple_of(scale_height * height, min_val=input_h)
    new_width = constrain_to_multiple_of(scale_width * width, min_val=input_w)

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
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    input_h, input_w = 518, 518 # 700, 700

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_img = cv2.imread(image_path)
    h, w = raw_img.shape[:2]
    print(f'original shape : {raw_img.shape}')
    raw_img = cv2.resize(raw_img, (input_w, input_h))

    input_image = preprocess_image(raw_img, input_h, input_w)  # Preprocess image
    print(f'after preprocess shape : {input_image.shape}')
    batch_images = np.concatenate([input_image], axis=0)

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    encoder = 'small'   # 'large' or 'base' or 'small' or 'Large-2w-iter'
    dynamo = False      # True or False
    onnx_sim = True    # True or False
    model_name = f"distill_any_depth_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes 
    input_shape = (batch_images.shape)
    output_shape = (1, batch_images.shape[2], batch_images.shape[3])
    print(f'trt input shape : {input_shape}')
    print(f'trt output shape : {output_shape}')

    iteration = 100
    dur_time = 0
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
        inputs[0].host = batch_images
                
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
        # ===================================================================

        print('[MDET] Post process')
        depth = torch.from_numpy(trt_outputs[0].reshape(output_shape))
        depth = torch.squeeze(depth).numpy()
        
        # Results
        print(f'[MDET] {iteration} iterations time: {dur_time:.4f} [sec]')
        avg_time = dur_time / iteration
        print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
        print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')
        print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')
    
    # ===================================================================
    print('[MDET] Generate color depth image')

    # visualization
    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt.jpg')

    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)

    color_depth = (cmap(depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    color_depth_bgr = cv2.resize(color_depth_bgr, (w, h), cv2.INTER_LINEAR)

    # save colored depth image 
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt')
    np.savez_compressed(output_file_npz, depth=depth)

    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()
