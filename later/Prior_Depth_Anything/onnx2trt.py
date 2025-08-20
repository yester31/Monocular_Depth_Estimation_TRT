# by yhpark 2025-8-1
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
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

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

def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    input_h, input_w = 518, 518 # 700, 700
    target_size = 1024
    original_coords = []  # Renamed from position_info to be more descriptive

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    print('[MDET] Pre process')
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height, width}")
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
    rgb = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    # Convert to tensor
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float() / 255.0
    rgb = rgb.unsqueeze(0)
    batch_images = F.interpolate(rgb, size=(input_h, input_w), mode="bilinear", align_corners=False)
    batch_images = batch_images.unsqueeze(0)
    batch_images = batch_images.cpu().numpy()

    # Model and engine paths
    onnx_dtype_fp16 = True
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    model_name = f"vggt_only_depth_{input_h}x{input_w}"
    model_name = f"{model_name}_fp16" if onnx_dtype_fp16 else model_name    
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', model_name, f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes 
    input_shape = (batch_images.shape)
    pose_enc_shape = (1, 1, 9)
    depth_shape = (1, input_h, input_w)
    depth_conf_shape = (1, input_h, input_w)

    print(f'[MDET] input shape : {input_shape}')
    print(f'[MDET] pose_enc shape : {pose_enc_shape}')
    print(f'[MDET] depth  shape : {depth_shape}')
    print(f'[MDET] depth_conf  shape : {depth_conf_shape}')

    iteration = 100
    dur_time = 0
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = batch_images
                
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
        # ===================================================================

        print('[MDET] Post process')
        depth = trt_outputs[0].reshape(depth_shape)
        depth = np.squeeze(depth)

        original_coord = original_coords[0]
        depth = depth[int(original_coord[1]/2) : int(original_coord[3]/2), :]

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

    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu + 1e-6)

    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    color_depth_bgr = cv2.resize(color_depth_bgr, (int(original_coord[4]), int(original_coord[5])), cv2.INTER_LINEAR)

    # save colored depth image 
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt')
    np.savez_compressed(output_file_npz, depth=depth)

    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()
