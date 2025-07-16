# by yhpark 2025-7-16
# Depth Pro TensorRT model generation
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import tensorrt as trt
import torch
from torch import nn

from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)

from PIL import Image
from matplotlib import pyplot as plt

import cv2
import os
import numpy as np
import time
import common
from common import *

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
print(f"current file path: {current_file_path}")
print(f"current directory: {current_directory}")

# Global Variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO

timing_cache = f"{current_directory}/timing.cache"

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
        build_time_str = f"{build_time} [sec]" if build_time < 60 else f"{build_time // 60} [min] {build_time % 60} [sec]"
        print(f'[TRT_E] engine build done! ({build_time_str})')

        return engine
    
def preprocess_image(img, precision= torch.float32):
   """
   Function to preprocess the image.
   Includes color conversion, tensor transformation, and normalization.

   Parameters:
      img (np.ndarray): Input image in BGR format.

   Returns:
      np.ndarray: Preprocessed image tensor.
   """
   transform = Compose(
        [
            ToTensor(),
            Lambda(lambda x: x.to(device)),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ConvertImageDtype(precision),
        ]
    )
   with torch.no_grad():
        img = np.array(img)
        tensor = transform(img).unsqueeze(0)
        tensor_resized = nn.functional.interpolate(
            tensor, size=(1536, 1536), mode='bilinear', align_corners=False
        )

   return tensor_resized.cpu().numpy()

def main():
    iteration = 20
    dur_time = 0

    # Input
    f_px = None
    image_file_name = 'example.jpg'
    image_path = os.path.join(current_directory, 'ml-depth-pro', 'data', image_file_name)
    img = Image.open(image_path)
    H = img.size[1]
    W = img.size[0]
    print(f'original shape : {img.size}')
    input_image = preprocess_image(img, torch.half)  # Preprocess image
    print(f'after preprocess shape : {input_image.size}')
    batch_images = np.concatenate([input_image], axis=0)
    print(f'trt input shape : {batch_images.shape}')

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    model_name = "dinov2l16_384"
    onnx_model_path = os.path.join(current_directory, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(current_directory, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # Output shapes 
    output_shapes = (1, 1, input_image.shape[2], input_image.shape[3])
    print(f'trt output shape : {output_shapes}')

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
                
        # inspector = engine.create_engine_inspector()
        # inspector.execution_context = context # OPTIONAL
        # print(inspector.get_layer_information(0, trt.tensorrt.LayerInformationFormat.JSON)) # Print the information of the first layer in the engine.
        # print(inspector.get_engine_information( trt.tensorrt.LayerInformationFormat.JSON)) # Print the information of the entire engine.
        
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = batch_images
        
        context.set_input_shape('input', batch_images.shape)
        
        # Warm-up      
        for _ in range(5):  
            common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        # Inference loop
        for _ in range(iteration):
            begin = time.time()
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            
            # # Reshape output
            canonical_inverse_depth = torch.from_numpy(trt_outputs[0].reshape(output_shapes))
            fov_deg = torch.from_numpy(trt_outputs[1])
            if f_px is None:
                f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))

            inverse_depth = canonical_inverse_depth * (W / f_px)
            inverse_depth = nn.functional.interpolate(
                inverse_depth, size=(H, W), mode='bilinear', align_corners=False
            )
            depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
            
            torch.cuda.synchronize()
            dur_time += time.time() - begin

    # Results
    print(f'[TRT] {iteration} iterations time: {dur_time:.4f} [sec]')
    avg_time = dur_time / iteration
    print(f'[TRT] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT] Average inference time: {avg_time * 1000:.2f} [msec]')
    print(f'[TRT] focal_length : {f_px}') # tensor([2938.7341])

    # post process
    depth = depth.numpy().squeeze()
    inverse_depth = 1 / depth
    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )

    if 0 : # prev version 
        #activation_map = (inverse_depth - np.min(inverse_depth)) / np.max(inverse_depth)
        heat_map = cv2.applyColorMap(np.uint8(255 * inverse_depth_normalized), cv2.COLORMAP_TURBO) # hw -> hwc
        # 샘플 결과 출력 및 저장
        save_path = os.path.join(current_directory, 'results', f'{os.path.splitext(image_file_name)[0]}_TRT_cv.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, heat_map)
    else : # original ml-pro 
        # Save as color-mapped "turbo" jpg image.
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
        # 샘플 결과 출력 및 저장
        save_path = os.path.join(current_directory, 'results', f'{os.path.splitext(image_file_name)[0]}_TRT.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(color_depth).save(save_path, format="JPEG", quality=90)

    output_file_npz = os.path.join(current_directory, 'results', f'{os.path.splitext(image_file_name)[0]}_TRT')
    np.savez_compressed(output_file_npz, depth=depth)

    common.free_buffers(inputs, outputs, stream)


if __name__ == '__main__':
    main()
