# by yhpark 2025-7-16
# Depth Pro TensorRT model generation
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
import common
from common import *
from core import bench

import torch
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)

from matplotlib import pyplot as plt
import cv2
import numpy as np
import time

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO

    
def main():

    # 100/20 like every other model. This used to be 20 iterations, too few
    # for a meaningful p99 -- at ~1536x1536 the run still finishes in seconds.
    iteration = 100
    warmup = 20
    img_size = 1536
    interpolation_mode = "bilinear"

    # Input
    f_px0 = None
    image_file_name = 'example.jpg'
    # image_file_name = '1.png'
    # image_file_name = 'panda0.jpg'
    image_path = os.path.join(_R, 'data', image_file_name)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'[MDET] original shape : {image_rgb.shape}')

    transform = Compose([
            ToTensor(), 
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = transform(image_rgb)
    
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    _, _, H, W = x.shape
    resize = H != img_size or W != img_size

    if resize:
        x = torch.nn.functional.interpolate(
            x, 
            size=(img_size, img_size), 
            mode=interpolation_mode, 
            align_corners=False
        )
    x = x.cpu().numpy()
    print(f'[MDET] model input size : {x.shape}') # [1, 3, 1536, 1536]

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    dynamo = True # True or False
    # Must match onnx_export.py, which builds the same name from the model's
    # own img_size. A mismatch now fails as a missing file instead of a
    # silently wrong engine.
    model_name = f"depth_pro_{img_size}x{img_size}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # Output shapes 
    output_shapes = (1, 1, img_size, img_size)
    print(f'[MDET] trt output shape : {output_shapes}')

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        inputs[0].host = x

        # Warm-up and timed loop both live in core/bench.py so every model
        # measures the same thing. warmup is 20 everywhere now; it used to vary
        # between 5 and 20, which alone made two models' numbers incomparable.
        #
        # The post-process below used to sit INSIDE this loop, so depth_pro's
        # published figure was inference plus a CPU torch pass that includes an
        # interpolate from 1536x1536 up to the source resolution. No other model
        # timed its post-process, which made depth_pro look far slower than it
        # is. It now runs once, after timing, like everywhere else.
        trt_outputs, samples = bench.measure(
            lambda: common.do_inference(context, engine=engine, bindings=bindings,
                                        inputs=inputs, outputs=outputs, stream=stream),
            warmup=warmup, iterations=iteration)

        # ===================================================================
        print('[MDET] Post process')
        canonical_inverse_depth = torch.from_numpy(trt_outputs[0].reshape(output_shapes))
        fov_deg = torch.from_numpy(trt_outputs[1])

        if f_px0 is None:
            f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        else :
            f_px = f_px0

        inverse_depth = canonical_inverse_depth * (W / f_px)
        f_px = f_px.squeeze()

        if resize:
            inverse_depth = torch.nn.functional.interpolate(
                inverse_depth, size=(H, W), mode=interpolation_mode, align_corners=False
            )

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

        # Results - printed as before, and written to reports/bench/ so
        # compare.py can build the table without anyone retyping a number.
        bench.record('depth_pro', samples, warmup=warmup, precision=precision,
                     profile='native', input_h=img_size, input_w=img_size,
                     engine_path=engine_file_path,
                     outputs={'depth': depth, 'f_px': f_px},
                     notes='1536x1536 is upstream fixed; no 518 bench profile exists',
                     model_input=x)

    common.free_buffers(inputs, outputs, stream)

    # post process
    depth = torch.squeeze(depth).numpy()
    print(f'[MDET] max : {depth.max()} , min : {depth.min()}')
    if f_px0 is not None:
        print(f"[MDET] focal length (from exif): {f_px:0.2f}")
    else :
        print(f"[MDET] predicted Focal length (by Depth Pro) : {f_px:0.2f}")


    print('[MDET] Generate color depth image')
    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )

    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)
    #color_depth_bgr = cv2.applyColorMap(np.uint8(255 * inverse_depth_normalized), cv2.COLORMAP_TURBO) # hw -> hwc

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)
    
    # save colored depth image 
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_trt.jpg')
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, f'{os.path.splitext(image_file_name)[0]}_trt')
    np.savez_compressed(output_file_npz, depth=depth)
    
    # save colored depth image with color depth bar
    output_file_depth_bar = os.path.join(save_dir_path, f'{os.path.splitext(image_file_name)[0]}_trt_depth_bar.jpg')
    plt.figure(figsize=(8, 6))
    img = plt.imshow(inverse_depth_normalized, cmap='turbo')  
    plt.axis('off')
    cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
    num_ticks = 5
    cbar_ticks = np.linspace(0, 1, num_ticks)
    cbar_ticklabels = np.linspace(depth.max(), depth.min(),  num_ticks)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{v:.2f} m' for v in cbar_ticklabels])
    cbar.set_label('Depth (m)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file_depth_bar, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    
if __name__ == '__main__':
    main()
