# by yhpark 2025-7-29
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import tensorrt as trt
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from matplotlib import pyplot as plt

import cv2
import numpy as np
import time
import common
from common import *
from core import bench

from UniK3D.unik3d.models.unik3d import get_paddings, get_resize_factor


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO


def postprocess_intrinsics(K, resize_factors, paddings = None):
    batch_size = K.shape[0]
    K_new = K.clone()

    for i in range(batch_size):
        scale_w, scale_h = resize_factors
        #pad_l, _, pad_t, _ = paddings[i]

        K_new[i, 0, 0] *= scale_w  # fx
        K_new[i, 1, 1] *= scale_h  # fy
        K_new[i, 0, 2] *= scale_w  # cx
        K_new[i, 1, 2] *= scale_h  # cy

        #K_new[i, 0, 2] -= pad_l  # cx
        #K_new[i, 1, 2] -= pad_t  # cy

    return K_new

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # 518x518, the size every model in this repo uses so their speeds compare.
    #
    # WARNING: the metric depth from this engine is NOT comparable to upstream.
    #
    # unik3d picks its own input size from the image (see the commented block
    # below): aspect preserved, pixel count clamped to 200k-600k, multiple of
    # 14. Given the original 3024x2268 it chooses 896x672. Pre-resizing to
    # 518x518 does not hand it a smaller image — it tells the model "518x518 is
    # the native resolution", and the model infers focal length from that.
    # Since it feeds that estimate straight into depth, the metric values move:
    #
    #   fed to model    metric scale vs upstream   AbsRel   structure corr
    #   896x672                          1.18x      17.9%          0.9961
    #   518x700                          1.75x      75.6%          0.8968
    #   518x518 (here)                   3.15x     217.5%          0.7210
    #
    # Measured on RTX 3080 with data/example.jpg. Note the structure survives —
    # after dividing out the scale the error is 5.6% — so relative depth is
    # usable; only the absolute metres are wrong.
    #
    # Not corrected here, because there is no fixed size that would fix it: the
    # model's choice depends on the source aspect ratio, so a static engine
    # would need one build per aspect. Use PyTorch when the metric values
    # matter. depth_anything_v2, depth_pro and moge_2 were checked and do not
    # behave this way (scale stays within 1.05x).
    input_h = 518 # 1036
    input_w = 518 # 1386

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    h, w = raw_image.shape[:2]
    resize_factors = (w/input_w, h/input_h)

    print(f'[MDET] original shape : {raw_image.shape}')
    raw_image = cv2.resize(raw_image, (input_w, input_h))
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32) 
    rgb = torch.from_numpy(image).permute(2, 0, 1) # C, H, W
    rgb = rgb.unsqueeze(0)
    rgb = TF.normalize(rgb.float() / 255.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)
    # What upstream unik3d.infer() does instead of the plain resize above.
    # Deliberately not used: new_H/new_W depend on the source aspect ratio, so
    # the input shape would vary per image and a static TensorRT engine cannot
    # accept that. This is the cost recorded in the WARNING at the top of main().
    '''
        B, _, H, W = rgb.shape
        ratio_bounds =  [0.5, 2.5]
        pixels_bounds= [200000, 600000]
        paddings, (padded_H, padded_W) = get_paddings((H, W), ratio_bounds)
        (pad_left, pad_right, pad_top, pad_bottom) = paddings
        resize_factor, (new_H, new_W) = get_resize_factor((padded_H, padded_W), pixels_bounds)
        rgb = F.pad(rgb, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        rgb = F.interpolate(rgb, size=(new_H, new_W), mode="bilinear", align_corners=False)
    '''
    x = rgb.cpu().numpy()
    print(f'[MDET] after preprocess shape : {x.shape}')
    batch_images = np.concatenate([x], axis=0)

    # Model and engine paths
    precision = "fp16"  # 'fp32' or 'fp16'
    encoder = 'vitb'    # 'vits' or 'vitb'  or 'vitl' 
    dynamic = False      # False
    onnx_sim = False     # True or False
    model_name = f"unik3d_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes 
    input_shape = (batch_images.shape)
    output_shape = (1, 3, batch_images.shape[2], batch_images.shape[3])
    print(f'[MDET] trt input shape : {input_shape}')
    print(f'[MDET] trt output shape : {output_shape}')
    if dynamic:
        dynamic_input_shapes = [[1,3,518,518], [1,3,518,518], [1,3,518,518]]
        dynamic_input_shapes = [[1,3,280,280], [1,3,518,518], [1,3,686,686]]
    else:
        dynamic_input_shapes = None

    iteration = 100
    warmup = 20
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision, dynamic_input_shapes) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
        inputs[0].host = batch_images
        if dynamic:
            context.set_input_shape('rgbs', batch_images.shape)

        # Warm-up and timed loop both live in core/bench.py so every model
        # measures the same thing. warmup is 20 everywhere now; it used to vary
        # between 5 and 20, which alone made two models' numbers incomparable.
        trt_outputs, samples = bench.measure(
            lambda: common.do_inference(context, engine=engine, bindings=bindings,
                                        inputs=inputs, outputs=outputs, stream=stream),
            warmup=warmup, iterations=iteration)

        # ===================================================================
        print('[MDET] Post process')
        points = torch.from_numpy(trt_outputs[0].reshape(output_shape))
        points = F.interpolate(points, (h, w), mode="bilinear", align_corners=False)
        depth = points[:, -1:]
        depth = torch.clamp(depth, min=1e-3, max=1e3)
        depth = torch.squeeze(depth).numpy()
        points = torch.squeeze(points).numpy()

        # Results - printed as before, and written to reports/bench/ so
        # compare.py can build the table without anyone retyping a number.
        bench.record('unik3d', samples, warmup=warmup, precision=precision,
                     profile='bench', input_h=input_h, input_w=input_w,
                     engine_path=engine_file_path, outputs={'depth': depth})
        print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')

    # ===================================================================
    print('[MDET] Generate color depth image')
    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu + 1e-6)

    # visualization
    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt.jpg')
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)

    # save colored depth image 
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt')
    np.savez_compressed(output_file_npz, depth=depth)

    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()