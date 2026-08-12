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
from core import bench

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def pre_process(raw_image, input_h, input_w):
    """Pad to a square, then resize once to the network size.

    This used to pad, resize to target_size=1024, and only then downscale to
    518, following upstream's load_and_preprocess_images_square(1024). But that
    helper exists to feed a 1024 network; here the engine is 518, so the extra
    hop only costs a full 1024x1024 cubic resize per frame and loses detail
    through a non-antialiased bilinear downscale. Removing it also puts the
    coordinates on the same grid as the output, so the crop needs no rescaling.
    """
    original_coords = []  # Renamed from position_info to be more descriptive
    height, width = raw_image.shape[:2]
    img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    # Make the image square by padding the shorter dimension
    max_dim = max(width, height)
    # Calculate padding
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    # Scale from the padded square to the network input
    scale = input_w / max_dim
    # Final coordinates of the original image inside the network input
    x1 = left * scale
    y1 = top * scale
    x2 = (left + width) * scale
    y2 = (top + height) * scale
    # Store original image coordinates and scale
    original_coords.append(np.array([x1, y1, x2, y2, width, height]))
    # Pad to a square. Upstream load_fn.py pads WHITE (value=1.0 on a 0-1
    # tensor); black changes what the network sees wherever the source is not
    # already square.
    padding = [255, 255, 255]
    img = cv2.copyMakeBorder(img, top, top, left, left, cv2.BORDER_CONSTANT, value=padding)

    img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)

    # Convert to tensor
    img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
    batch_images = img.unsqueeze(0).unsqueeze(0)
    return batch_images.cpu().numpy(), x1, y1, x2, y2


def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # Input
    input_h, input_w = 518, 518
    image_path = os.path.join(CUR_DIR, '..', 'data', 'example.jpg')
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    raw_image = cv2.imread(image_path)
    height, width = raw_image.shape[:2]
    print(f"[MDET] original image size : {height, width}")

    print('[MDET] Pre process')
    batch_images, x1, y1, x2, y2 = pre_process(raw_image, input_h, input_w)

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
    depth_shape = (1, input_h, input_w)
    print(f'[MDET] input shape : {input_shape}')
    print(f'[MDET] depth  shape : {depth_shape}')

    iteration = 100
    warmup = 20
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision, workspace_gib=4) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
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
        depth = trt_outputs[0].reshape(depth_shape)
        depth = np.squeeze(depth)

        # Results - printed as before, and written to reports/bench/ so
        # compare.py can build the table without anyone retyping a number.
        bench.record('vggt', samples, warmup=warmup, precision=precision,
                     profile='bench', input_h=input_h, input_w=input_w,
                     engine_path=engine_file_path, outputs={'depth': depth})
        print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')

    # ===================================================================
    print('[MDET] Generate color depth image')

    # visualization
    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    output_file_depth = os.path.join(save_dir_path, f"{image_file_name}_vggt_{input_h}x{input_w}_trt.jpg")

    # The coordinates are already on the output grid, so the upscale to 1024
    # that used to precede this crop is no longer needed.
    depth = depth[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2)), ...] 
    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu + 1e-6)

    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    color_depth_bgr = cv2.resize(color_depth_bgr, (width, height), cv2.INTER_LINEAR)

    # save colored depth image 
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, f"{image_file_name}_vggt_{input_h}x{input_w}_trt")
    np.savez_compressed(output_file_npz, depth=depth)

    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()
