# by yhpark 2025-8-2
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
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def main():
    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    input_h, input_w = 518, 518 # 700, 700
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

    # Resize straight to the network size. This used to go via target_size=1024
    # and then downscale to 518, copied from VGGT's
    # load_and_preprocess_images_square(target_size=1024). StreamVGGT upstream
    # has no such function -- its load_fn.py only ever uses 518 -- and the extra
    # hop cannot improve quality: upsampling then downsampling with a
    # non-antialiased bilinear only loses detail and costs a full 1024x1024
    # cubic resize per frame. It also left the coordinates on a 1024 grid while
    # the output was 518, which the postprocess then approximated by halving.
    scale = input_w / max_dim

    # Final coordinates of the original image inside the network input
    x1 = left * scale
    y1 = top * scale
    x2 = (left + width) * scale
    y2 = (top + height) * scale

    # Store original image coordinates and scale
    original_coords.append(np.array([x1, y1, x2, y2, width, height]))

    # Pad to a square. Upstream load_fn.py pads WHITE (value=1.0 on a 0-1
    # tensor); this was black, which is a different input to the network
    # wherever the source is not already square.
    padding = [255, 255, 255]
    img = cv2.copyMakeBorder(img, top, top, left, left, cv2.BORDER_CONSTANT, value=padding)

    rgb = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)

    # Convert to tensor
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float() / 255.0
    batch_images = rgb.unsqueeze(0).unsqueeze(0)
    batch_images = batch_images.cpu().numpy()

    # Model and engine paths
    onnx_dtype_fp16 = True
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    model_name = f"streamvggt_only_depth_{input_h}x{input_w}"
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
    # print(f'[MDET] pose_enc shape : {pose_enc_shape}')
    print(f'[MDET] depth  shape : {depth_shape}')
    # print(f'[MDET] depth_conf  shape : {depth_conf_shape}')

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

        # Crop the padding away. The coordinates are now on the same grid as
        # the output, so no rescaling is needed; previously they were computed
        # at 1024 and divided by 2 to reach 518, which is off by 1024/518/2 =
        # 1.2%. The horizontal crop was also missing entirely, so a landscape
        # source kept its left and right padding in the result.
        x1, y1, x2, y2 = original_coords[0][:4]
        depth = depth[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))]

        # Results - printed as before, and written to reports/bench/ so
        # compare.py can build the table without anyone retyping a number.
        bench.record('streamvggt', samples, warmup=warmup, precision=precision,
                     profile='bench', input_h=input_h, input_w=input_w,
                     engine_path=engine_file_path, outputs={'depth': depth})
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
    # original_coords, not original_coord — the singular name never existed, so
    # this line raised NameError after the whole inference had already run.
    # [4] and [5] are the source width and height recorded during preprocessing.
    color_depth_bgr = cv2.resize(
        color_depth_bgr,
        (int(original_coords[0][4]), int(original_coords[0][5])),
        interpolation=cv2.INTER_LINEAR)

    # save colored depth image 
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt')
    np.savez_compressed(output_file_npz, depth=depth)

    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()
