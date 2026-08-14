# by yhpark 2025-7-30
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
import torchvision.transforms.v2.functional as TF
from matplotlib import pyplot as plt

import cv2
import numpy as np
import time
from core import common
from core.common import *
from core import bench
from core import preprocess as pp

# Removed: `from unidepth.models.unidepthv2.unidepthv2 import get_paddings,
# get_resize_factor`. That is UniDepth's code in the Metric3D script, copied
# in and never called -- this file does its own keep-ratio resize and padding
# with cv2 below. It was a hard import, so it made this script require the
# UniDepth package to be installed in the shared trte environment in order to
# run at all.


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO

    

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    # WARNING: what comes out of here is canonical depth, NOT metric depth.
    #
    # Metric3D trains every dataset re-projected onto one virtual camera of
    # focal length 1000px. The network therefore predicts in that camera's
    # frame, and upstream converts back at the end:
    #
    #     canonical_to_real_scale = real_focal_length * scale / 1000.0
    #     pred_depth = pred_depth * canonical_to_real_scale
    #
    # That step is missing below, because real_focal_length is unknown for an
    # arbitrary image. infer.py has the same block behind `if 0:` with four
    # candidate values tried and abandoned, one of them borrowed from
    # depth_pro. Supply the real focal length (EXIF, calibration, or another
    # model's estimate) if you need metres.
    #
    # Measured consequence: the keep-ratio factor below moves 0.2716 -> 1.1892
    # across input sizes, a 4.4x swing, yet the output barely changes (scale
    # 1.00-1.12). Invariance to input size is exactly what "not yet tied to a
    # real camera" looks like. See docs/model_contracts.md D12 and 5.7.
    #
    # 616x1064 is upstream's own size for the ViT models, so this script is
    # already native; it does not use the repo-wide 518 bench size.
    input_h = 616
    input_w = 1064

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(_R, 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    print(f'[MDET] original shape : {raw_image.shape}')

    # This is the only model of the fourteen that keeps the aspect ratio and
    # pads. core/preprocess.py reproduces both of its quirks rather than tidying
    # them: the resized size truncates (int(w * scale), not round), and there is
    # no normalisation at all -- Metric3DExportModel.forward() (upstream
    # onnx/metric3d_onnx_export.py) calls normalize_image() as its first step,
    # so mean/std are baked into the graph and it expects raw 0-255 input.
    # Normalising here as well would apply the statistics twice. The padding is
    # the ImageNet mean colour in those same 0-255 units.
    #
    # tests/test_preprocess.py asserts np.array_equal against
    # reports/inputs/metric3d_v2.npy.
    batch_images, geom = pp.preprocess_for(raw_image, 'metric3d_v2',
                                           (input_h, input_w))
    # Same four numbers the inline code produced, now derived from the geometry
    # preprocessing recorded, so the crop below cannot drift from the pad above.
    pad_info = [geom.pad_top, geom.pad_bottom, geom.pad_left, geom.pad_right]

    print(f'[MDET] after preprocess shape : {batch_images.shape}')

    # Model and engine paths (ff)
    precision = "fp32"  # 'fp32'
    encoder = 'vits'    # 'vits' or vitl or vitg
    onnx_sim = False     # True or False
    model_name = f"metric3d_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    #onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'model.onnx')
    #engine_file_path = os.path.join(CUR_DIR, 'engine', f'model_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes 
    input_shape = (batch_images.shape)
    output_shape = (1, 1, batch_images.shape[2], batch_images.shape[3])
    print(f'[MDET] trt input shape : {input_shape}')
    print(f'[MDET] trt output shape : {output_shape}')
    # dynamic_input_shapes = [[1,3,616, 1064], [1,3,616, 1064], [1,3,616, 1064]]
    dynamic_input_shapes = None

    iteration = 100
    warmup = 20
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision, dynamic_input_shapes, obey_precision_constraints=True) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
        inputs[0].host = batch_images
        
        #context.set_input_shape('input', batch_images.shape)

        # Warm-up and timed loop both live in core/bench.py so every model
        # measures the same thing. warmup is 20 everywhere now; it used to vary
        # between 5 and 20, which alone made two models' numbers incomparable.
        trt_outputs, samples = bench.measure(
            lambda: common.do_inference(context, engine=engine, bindings=bindings,
                                        inputs=inputs, outputs=outputs, stream=stream),
            warmup=warmup, iterations=iteration)

        # ===================================================================
        print('[MDET] Post process')
        pred_depth = torch.from_numpy(trt_outputs[0].reshape(output_shape)).squeeze()
        
        # Results - printed as before, and written to reports/bench/ so
        # compare.py can build the table without anyone retyping a number.
        bench.record('metric3d_v2', samples, encoder=encoder, warmup=warmup, precision=precision,
                     profile='native', input_h=input_h, input_w=input_w,
                     engine_path=engine_file_path, outputs={'depth': pred_depth},
                     model_input=batch_images)
    # ===================================================================
    print('[MDET] Post process')
    # Metric Depth Estimation
    # un pad
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :],
                                                 (geom.src_h, geom.src_w),
                                                 mode='bilinear').squeeze()
    ###################### canonical camera space ######################
    # Still canonical here. The de-canonical transform would go on this line;
    # see the WARNING at the top of main() for why it does not.
    pred_depth = torch.clamp(pred_depth, 0, 300)
    pred_depth = pred_depth.numpy()
    print(f'[MDET] max : {pred_depth.max():0.5f} , min : {pred_depth.min():0.5f}')
    # ===================================================================
    print('[MDET] Generate color depth image')
    # visualization
    inverse_depth = 1 / pred_depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu + 1e-6)

    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    

    # save colored depth image 
    output_file_depth = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + f'_{encoder}_{precision}_trt3.jpg')
    cv2.imwrite(output_file_depth, color_depth_bgr)

    # save_npz
    # output_file_npz = os.path.join(save_dir_path, os.path.splitext(image_file_name)[0]+ f'_{encoder}_{precision}_trt2')
    # np.savez_compressed(output_file_npz, depth=pred_depth)

    common.free_buffers(inputs, outputs, stream)

if __name__ == '__main__':
    main()