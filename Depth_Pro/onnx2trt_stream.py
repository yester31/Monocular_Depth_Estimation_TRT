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
        build_time_str = f"{build_time:.2f} [sec]" if build_time < 60 else f"{build_time // 60 :.2f} [min] {build_time % 60 :.2f} [sec]"
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
            #Lambda(lambda x: x.to(device)),
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
    count = 0
    dur_time = 0
    f_px = None

    # Input
    cap = cv2.VideoCapture('http://192.168.0.11:5000/video')
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"original shape : {original_width} x {original_height}")

    input_shape = (1, 3, 1536, 1536)
    output_shape = (1, 1, 1536, 1536)
    print(f'trt input shape : {input_shape}')
    print(f'trt output shape : {output_shape}')

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    model_name = "dinov2l16_384"
    onnx_model_path = os.path.join(current_directory, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(current_directory, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)


    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        context.set_input_shape('input', input_shape)
        
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            begin = time.time()
            # pre-proc
            inputs[0].host = preprocess_image(frame, torch.half)  # Preprocess image
            # infer
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            
            # post process
            canonical_inverse_depth = torch.from_numpy(trt_outputs[0].reshape(output_shape))
            fov_deg = torch.from_numpy(trt_outputs[1])
            if f_px is None:
                f_px = 0.5 * original_width / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))

            inverse_depth = canonical_inverse_depth * (original_width / f_px)
            inverse_depth = nn.functional.interpolate(inverse_depth, size=(original_height, original_width), mode='bilinear', align_corners=False)
            inverse_depth = torch.clamp(inverse_depth, min=1e-4, max=1e4).numpy().squeeze()

            dur_time += time.time() - begin
            count += 1

            # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
            max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
            min_invdepth_vizu = max(1 / 250, inverse_depth.min())
            inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)
            heat_map = cv2.applyColorMap(np.uint8(255 * inverse_depth_normalized), cv2.COLORMAP_TURBO) # hw -> hwc

            # 영상에 FPS 표시
            fps = count / (dur_time + 1e-6)
            cv2.putText(heat_map, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 캡처된 프레임을 윈도우에 표시합니다.
            cv2.imshow('Webcam Stream (Depth Pro)', heat_map)

            # 'q' 키를 누르면 반복문에서 빠져나옵니다.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Results
    print(f'[TRT] {count} iterations time: {dur_time:.4f} [sec]')
    avg_time = dur_time / count
    print(f'[TRT] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[TRT] Average inference time: {avg_time * 1000:.2f} [msec]')
    print(f'[TRT] focal_length : {f_px}') 
    
    common.free_buffers(inputs, outputs, stream)


if __name__ == '__main__':
    main()
