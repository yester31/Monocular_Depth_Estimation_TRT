# by yhpark 2025-7-28
# Depth Anything AC TensorRT model generation
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

    
# preprocess_image() used to live here and is now core/preprocess.py. This is
# the model that pins the keep-ratio type: its ceil-to-a-multiple-of-14 rule
# differs from depth_anything_v2's constrain rule (700 against 686 for a 4:3
# frame at target 518), and it is the only script whose second profile leaves
# that rule with real work to do.
#
# Its arithmetic is also the reason core/preprocess.py carries dtypes: the /255
# happens in float32 here and in float64 in its two siblings, and the ImageNet
# statistics then promote back to float64 in all three. tests/test_preprocess.py
# asserts np.array_equal against reports/inputs/depth_anything_ac.npy.


def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # Profile. Must match onnx_export.py — the engine input shape is static.
    #
    #   bench   518x518. Every model in this repo runs at 518x518 so their
    #           speeds are comparable. Reaching a square from a 4:3 source
    #           means stretching, which upstream never does.
    #   native  what upstream's tools/infer.py produces: short side 518, aspect
    #           preserved, rounded up to a multiple of 14. Its exact size
    #           depends on the source aspect ratio, so the engine is pinned to
    #           the 4:3 of data/example.jpg -> 518x700.
    #
    # Measured difference between the two on data/example.jpg: 6.03%.
    profile = 'bench'   # 'bench' or 'native'
    input_h, input_w = (518, 518) if profile == 'bench' else (518, 700)

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(_R, 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    print(f'[MDET] original shape : {raw_image.shape}')
    print(f'[MDET] profile : {profile} -> {input_h}x{input_w}')

    # stretch=False is the `native` profile: it hands the source through
    # untouched so the keep-ratio rule decides the size, which for a 4:3 frame
    # at target 518 is 518x700. 'bench' squashes to the square first and the
    # keep-ratio stage then has nothing left to do.
    batch_images, geom = pp.preprocess_for(raw_image, 'depth_anything_ac',
                                           (input_h, input_w),
                                           stretch=(profile == 'bench'))
    print(f'[MDET] after preprocess shape : {batch_images.shape}')

    # Model and engine paths (ff)
    precision = "fp16"  # 'fp32' or 'fp16'
    encoder = 'vits'    # 'vits'
    dynamo = True       # True or False
    # Measured 2026-08-13: simplified 4.37 ms against 10.03 unsimplified.
    # Simplification folds MatMul+Add into Gemm and TensorRT maps Gemm to
    # its fp16 kernels, so the transformer body stopped running in fp32
    # (86.4% of the time was in Float layers before). Its two siblings,
    # depth_anything_v2 and distill_any_depth, were already built this way
    # and sit at 4.30 and 4.26 ms.
    onnx_sim = True
    # The resolution in the name already distinguishes the two profiles, so
    # bench and native engines coexist without overwriting each other.
    model_name = f"depth_anything_AC_{encoder}_{input_h}x{input_w}"
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
        # geom.src_* is the size preprocessing recorded, so the map goes back to
        # the grid it came from rather than to a separately-read shape.
        depth = F.interpolate(depth[:, None], (geom.src_h, geom.src_w),
                              mode="bilinear", align_corners=True)[0, 0]
        depth = torch.clamp(depth, min=1e-3, max=1e3)
        depth = torch.squeeze(depth).numpy()

        # Results - printed as before, and written to reports/bench/ so
        # compare.py can build the table without anyone retyping a number.
        bench.record('depth_anything_ac', samples, encoder=encoder, warmup=warmup,
                     precision=precision, profile=profile, input_h=input_h,
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
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    color_depth = (cmap(depth_normalized)[..., :3] * 255).astype(np.uint8)

    # save colored depth image 
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{model_name}_trt')
    np.savez_compressed(output_file_npz, depth=depth)

    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()