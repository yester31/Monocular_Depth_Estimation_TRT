# by yhpark 2025-7-27
# Distill Any Depth TensorRT model generation
import os
import sys
# Repo root, found by walking up to the directory holding core/ rather
# than counting "..". Counting breaks the moment a script moves, and
# this one has moved: Phase 4 put every model under models/.
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "core")) and os.path.dirname(_R) != _R:
    _R = os.path.dirname(_R)
sys.path.insert(1, _R)

import tensorrt as trt
import torch
import torch.nn.functional as F

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib

import cv2
import numpy as np
import time
from core import common
from core.common import *
from core import bench
from core import preprocess as pp

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO


# constrain_to_multiple_of() and preprocess_image() are now core/preprocess.py.
# Worth naming what this model's second stage actually was: it derives a scale
# per axis and snaps each independently, so unlike depth_anything_v2 it does not
# preserve the aspect ratio -- it is a stretch with the sides rounded to 14.
# core/preprocess.py calls that type `snap` rather than folding it into
# keep_ratio, because at a size that is not already a multiple of 14 the two
# disagree. tests/test_preprocess.py asserts np.array_equal against
# reports/inputs/distill_any_depth.npy.


def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    input_h, input_w = 518, 518 # 700, 700

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(_R, 'data', image_file_name)
    raw_img = cv2.imread(image_path)
    print(f'original shape : {raw_img.shape}')

    batch_images, geom = pp.preprocess_for(raw_img, 'distill_any_depth',
                                           (input_h, input_w))
    print(f'after preprocess shape : {batch_images.shape}')

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    encoder = 'small'   # 'large' or 'base' or 'small' or 'Large-2w-iter'
    dynamo = True      # True or False
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
    warmup = 20
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
        inputs[0].host = batch_images
                
        # Warm-up and timed loop both live in core/bench.py so every model
        # measures the same thing. warmup is 20 everywhere now; it used to vary
        # between 5 and 20, which alone made two models' numbers incomparable.
        trt_outputs, samples = bench.measure(
            lambda: common.do_inference(context, engine=engine, bindings=bindings,
                                        inputs=inputs, outputs=outputs, stream=stream),
            warmup=warmup, iterations=iteration)
        # ===================================================================

        print('[MDET] Post process')
        depth = torch.from_numpy(trt_outputs[0].reshape(output_shape))
        depth = torch.squeeze(depth).numpy()
        
        # Results - printed as before, and written to reports/bench/ so
        # compare.py can build the table without anyone retyping a number.
        bench.record('distill_any_depth', samples, encoder=encoder, warmup=warmup,
                     precision=precision, profile='bench', input_h=input_h,
                     input_w=input_w, engine_path=engine_file_path,
                     outputs={'depth': depth},
                     model_input=batch_images)
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
    color_depth_bgr = cv2.resize(color_depth_bgr, (geom.src_w, geom.src_h),
                                 cv2.INTER_LINEAR)

    # save colored depth image 
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt')
    np.savez_compressed(output_file_npz, depth=depth)

    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()
