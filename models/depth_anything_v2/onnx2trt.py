# by yhpark 2025-7-26
# Depth Anything V2 TensorRT model generation
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
from matplotlib import pyplot as plt

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


# constrain_to_multiple_of() and preprocess_image() used to live here. They are
# now core/preprocess.py -- the squash to the engine's size, the /255 in
# float64, upstream's constrain-to-multiple-of-14 rule and the ImageNet
# statistics, in that order and those dtypes. Nothing about the arithmetic
# changed: tests/test_preprocess.py asserts np.array_equal against
# reports/inputs/depth_anything_v2.npy, which is the tensor the engine this
# repository's numbers were measured on actually received.


def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    input_h = 518 # 1036
    input_w = 518 # 1386

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(_R, 'data', image_file_name)
    raw_img = cv2.imread(image_path)
    print(f'[MDET] original shape : {raw_img.shape}')

    batch_images, geom = pp.preprocess_for(raw_img, 'depth_anything_v2',
                                           (input_h, input_w))
    print(f'[MDET] after preprocess shape : {batch_images.shape}')

    # Model and engine paths
    precision = "fp16"  # 'fp32' or 'fp16'
    encoder = 'vits'    # 'vits' or 'vitb' or 'vitg'
    metric_model = True # True or False
    dataset = 'hypersim'# 'hypersim' for indoor model, 'vkitti' for outdoor model
    dynamo = True       # True or False
    onnx_sim = True     # True or False
    dynamic = False     # fail...(False only)
    model_name = f"depth_anything_v2_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_metric_{dataset}" if metric_model else model_name
    model_name = f"{model_name}_dynamic" if dynamic else model_name
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

    if dynamic:
        #dynamic_input_shapes = [[1,3,280,280], [1,3,518,518], [1,3,686,686]]
        #dynamic_input_shapes = [[1,3,280,280], [1,3,518,686], [1,3,686,686]]
        dynamic_input_shapes = [[1,3,518,518], [1,3,518,518], [1,3,518,518]]
    else :
        dynamic_input_shapes = None

    iteration = 100
    warmup = 20
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision, dynamic_input_shapes) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
        inputs[0].host = batch_images
        
        if dynamic:
            context.set_input_shape('input', batch_images.shape)
        
        # Warm-up and timed loop both live in core/bench.py so every model
        # measures the same thing. See that module for what the number covers.
        trt_outputs, samples = bench.measure(
            lambda: common.do_inference(context, engine=engine, bindings=bindings,
                                        inputs=inputs, outputs=outputs, stream=stream),
            warmup=warmup, iterations=iteration)

        # ===================================================================
        print('[MDET] Post process')
        depth = torch.from_numpy(trt_outputs[0].reshape(output_shape))
        # geom.src_* is the size preprocessing recorded, so the map goes back to
        # the grid it came from rather than to a separately-read shape.
        depth = F.interpolate(depth[:, None], (geom.src_h, geom.src_w),
                              mode="bilinear", align_corners=True)[0, 0]
        depth = torch.clamp(depth, min=1e-3, max=1e3)
        depth = torch.squeeze(depth).numpy()

        # Results — printed as before, and written to reports/bench/ so
        # compare.py can build the table without anyone retyping a number.
        bench.record('depth_anything_v2', samples, warmup=warmup,
                     precision=precision, profile='bench',
                     input_h=input_h, input_w=input_w,
                     engine_path=engine_file_path, outputs={'depth': depth},
                     notes=f"encoder={encoder}, "
                           f"weights={'metric_' + dataset if metric_model else 'relative'}",
                     model_input=batch_images)
        print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')

    # ===================================================================
    print('[MDET] Generate color depth image')

    # visualization
    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt.jpg')
    if metric_model :
        inverse_depth = 1 / depth
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    else:
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(depth_normalized)[..., :3] * 255).astype(np.uint8)

    # save colored depth image 
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt')
    np.savez_compressed(output_file_npz, depth=depth)

    if metric_model :
        # save colored depth image with color depth bar
        output_file_depth_bar = os.path.join(save_dir_path, f'{os.path.splitext(image_file_name)[0]}_{model_name}_trt_depth_bar.jpg')
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
        
    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()