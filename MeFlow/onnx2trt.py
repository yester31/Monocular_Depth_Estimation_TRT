# by yhpark 2025-8-9
# MeFlow TensorRT model generation
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import tensorrt as trt
import torch
from PIL import Image
import cv2
import numpy as np
import time
import common
from common import *

sys.path.insert(1, os.path.join(sys.path[0], "MeFlow"))
from MeFlow.utils.flow_viz import flow_to_image

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

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
        print(f'[MDET] Engine build done! ({show_build_time(build_time)})')  
        return engine

def show_build_time(build_time):      
    if build_time < 60:
        build_time_str = f"{build_time:.2f} sec"
    elif build_time < 3600:
        minutes = int(build_time // 60)
        seconds = build_time % 60
        build_time_str = f"{minutes} min {seconds:.2f} sec"
    else:
        hours = int(build_time // 3600)
        minutes = int((build_time % 3600) // 60)
        seconds = build_time % 60
        build_time_str = f"{hours} hr {minutes} min {seconds:.2f} sec"
    return build_time_str
    
def load_image(image_path, new_size=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if new_size is not None :
        image = cv2.resize(image, new_size)
    # [H, W, C] -> [C, H, W]
    image = np.transpose(image, (2, 0, 1))
    # int - > float
    image = np.ascontiguousarray(image).astype(np.float32)
    # [C, H, W] -> [1, C, H, W]
    image = np.expand_dims(image, axis=0)
    return image

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results', 'tensorrt')
    os.makedirs(save_dir_path, exist_ok=True)

    input_h, input_w = 288, 512  # divisible by 8

    # Input
    image_dir_name = 'video_frames'
    image_dir = os.path.join(CUR_DIR, '..', image_dir_name)
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]
    # List all files in the directory and filter only image files
    image_paths = [
        os.path.join(image_dir, fname) for fname in sorted(os.listdir(image_dir)) if os.path.splitext(fname)[1].lower() in valid_exts
    ]

    # Model and engine paths
    precision = "fp16"  # 'fp32' or 'fp16'
    dynamo = True       # True or False
    onnx_sim = True     # True or False
    model_name = f"MeFlow_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes 
    output_shape = {
        "flow_low": (1, 2, int(input_h/8), int(input_w/8)),
        "flow_up": (1, 2, input_h, input_w)
        }
    print(f'[MDET] "flow_low" shape : {output_shape["flow_low"]}')
    print(f'[MDET] "flow_up" shape : {output_shape["flow_up"]}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{CUR_DIR}/results/output_video_trt.mp4", fourcc, 20, (640, 480))

    dur_time = 0
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        # Warm-up      
        for _ in range(10):  
            common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        # Inference loop
        for imfile1, imfile2 in zip(image_paths[:-1], image_paths[1:]):
            image1 = load_image(imfile1, (input_w, input_h))
            image2 = load_image(imfile2, (input_w, input_h))
            inputs[0].host = image1
            inputs[1].host = image2

            begin = time.time()
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

            flow_up = trt_outputs[1].reshape(output_shape["flow_up"])
            flow_up = np.transpose(flow_up[0], (1, 2, 0)) 
            flow_img = flow_to_image(flow_up) # map flow to rgb image
            flow_img_bgr = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR) 
            
            image1 = np.transpose(image1[0], (1, 2, 0)) 
            image1_bgr = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

            img_flo = np.concatenate([image1_bgr, flow_img_bgr], axis=0)

            cv2.imshow('image', img_flo/255.0)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            output_path = f'{save_dir_path}/{os.path.splitext(os.path.basename(imfile1))[0]}_optical_flow.jpg'
            cv2.imwrite(output_path, img_flo)

            frame = cv2.resize(flow_img_bgr, (640, 480))
            out.write(frame)

        out.release()
        cv2.destroyAllWindows()
        # ===================================================================
        # Results
        iteration = len(image_paths) - 1
        print(f'[MDET] {iteration} iterations time: {dur_time:.4f} [sec]')
        avg_time = dur_time / iteration
        print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
        print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')
    # ===================================================================
    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()