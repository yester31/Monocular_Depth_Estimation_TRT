# by yhpark 2025-7-28
# Depth Anything AC TensorRT model generation
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
                print(f'[MDET] set fp16 model')

            for i_idx in range(network.num_inputs):
                print(f'[MDET] input({i_idx}) name: {network.get_input(i_idx).name}, shape= {network.get_input(i_idx).shape}')
                
            for o_idx in range(network.num_outputs):
                print(f'[MDET] output({o_idx}) name: {network.get_output(o_idx).name}, shape= {network.get_output(o_idx).shape}')
    
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            common.save_timing_cache(config, timing_cache)
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            
            return engine

    if os.path.exists(engine_file_path):
        print(f"[MDET] Load engine from file ({engine_file_path})")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print(f'[MDET] Build engine ({engine_file_path})')
        begin = time.time()
        engine = build_engine()
        build_time = time.time() - begin
        build_time_str = f"{build_time:.2f} [sec]" if build_time < 60 else f"{build_time // 60 :.1f} [min] {build_time % 60 :.2f} [sec]"
        print(f'[MDET] Engine build done! ({build_time_str})')

        return engine
    
def preprocess_image(image, target_size=518):
    """Preprocess input image"""
    
    h, w = image.shape[:2]
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    new_h = ((new_h + 13) // 14) * 14
    new_w = ((new_w + 13) // 14) * 14
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # PrepareForNet
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image).astype(np.float32)

    # [C, H, W] -> [1, C, H, W]
    image = np.expand_dims(image, axis=0)

    return image

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    input_h = 518 # 1036
    input_w = 518 # 1386

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    h, w = raw_image.shape[:2]
    print(f'[MDET] original shape : {raw_image.shape}')
    raw_image = cv2.resize(raw_image, (input_w, input_h))
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    input_image = preprocess_image(image, input_h)  # Preprocess image
    print(f'[MDET] after preprocess shape : {input_image.shape}')
    batch_images = np.concatenate([input_image], axis=0)

    # Model and engine paths (ff)
    precision = "fp16"  # 'fp32' or 'fp16'
    encoder = 'vits'    # 'vits'
    dynamo = True       # True or False
    onnx_sim = False     # True or False
    model_name = f"depth_anything_AC_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes 
    input_shape = (batch_images.shape)
    output_shape = (1, batch_images.shape[2], batch_images.shape[3])
    print(f'[MDET] trt input shape : {input_shape}')
    print(f'[MDET] trt output shape : {output_shape}')

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
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        depth = torch.clamp(depth, min=1e-3, max=1e3)
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
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    color_depth = (cmap(depth_normalized)[..., :3] * 255).astype(np.uint8)

    # save colored depth image 
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt')
    np.savez_compressed(output_file_npz, depth=depth)

    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()