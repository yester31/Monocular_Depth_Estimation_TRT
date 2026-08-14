# by yhpark 2025-7-29
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

# get_paddings / get_resize_factor were imported here, but the only place they
# appear is the ''' ''' block in main() showing the upstream resize rule this
# script deliberately does not use. A hard import for a quoted example meant
# the shared trte environment had to have the whole UniDepth package installed
# before this script would run.


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

    # 672x896, which is the size this model chooses for itself.
    #
    # UniDepthV2 carries the same shape_constraints as unik3d -- aspect
    # preserved, pixel count clamped to 200k-600k, multiple of 14 -- and
    # resizes internally. For any 4:3 source, including every DIODE frame at
    # 1024x768, that rule lands on exactly 672x896.
    #
    # This used to be 518x518, to match the other models so the speeds would
    # compare, and that was the wrong trade. Pre-resizing tells the model that
    # 518x518 IS the native resolution, and it infers focal length accordingly:
    #
    #   fed to model      inferred fx
    #   original                2859.4
    #   518x700                  627.6
    #   518x518 (was)            551.1
    #
    # Metric depth is tied to focal length, so the depth moved with it: unik3d,
    # which shares this design, measured 3.15x upstream at 518x518. D11 was
    # never a conversion defect, it was the cost of overriding the model's own
    # choice.
    #
    # The objection that killed this before was that 896x672 is specific to one
    # sample image and a static engine would need one build per aspect ratio.
    # True in general, and not true of a fixed evaluation set: DIODE indoors is
    # 1024x768 throughout. See docs/model_contracts.md 5.7 -- depth_anything_v2,
    # depth_pro and moge_2 were checked and stay within 1.05x, so this is
    # specific to these two models.
    #
    # It costs speed: 602112 pixels against 268324 is 2.24x the work.
    input_h = 672
    input_w = 896

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(_R, 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    print(f'[MDET] original shape : {raw_image.shape}')
    print(f'[MDET] input size : {input_h}x{input_w}')
    # The stretch, the /255 and the ImageNet statistics are core/preprocess.py's
    # now. All three steps run in float32 here, matching torchvision's
    # TF.normalize on a float tensor bit for bit -- IEEE subtract and divide are
    # exact, so "matching" is byte equality and not a tolerance.
    # tests/test_preprocess.py asserts np.array_equal against
    # reports/inputs/unidepth_v2.npy.
    #
    # What upstream unidepth_v2.infer() does instead is pad the aspect ratio into
    # [0.5, 2.5] and scale the pixel count into [200000, 600000]. Deliberately
    # not used: new_H/new_W depend on the source aspect ratio, so the input shape
    # would vary per image and a static TensorRT engine cannot accept that. This
    # is the cost recorded in the WARNING at the top of main(). For a 4:3 source
    # the rule lands on 672x896, which is what is built.
    batch_images, geom = pp.preprocess_for(raw_image, 'unidepth_v2',
                                           (input_h, input_w))
    # The intrinsics rescale below needs how far the source was squashed, which
    # is the geometry preprocessing just recorded rather than a second reading
    # of the same two shapes.
    resize_factors = (geom.src_w / geom.dst_w, geom.src_h / geom.dst_h)
    print(f'[MDET] after preprocess shape : {batch_images.shape}')

    # Model and engine paths
    precision = "fp16"  # 'fp32' or 'fp16'
    encoder = 'vits'    # 'vits' or 'vitb'  or 'vitl' 
    dynamo = False      # False
    # Measured 2026-08-13: simplified 15.75 ms against 9.85 unsimplified --
    # 60% worse. Same fold as above, opposite result. Per-model, not a rule.
    onnx_sim = False
    model_name = f"uni_depth_v2_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes 
    input_shape = (batch_images.shape)
    output_shape = (1, 3, batch_images.shape[2], batch_images.shape[3])
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
        points = torch.from_numpy(trt_outputs[0].reshape(output_shape))
        points = F.interpolate(points, (geom.src_h, geom.src_w),
                               mode="bilinear", align_corners=False)
        depth = points[:, -1:]
        depth = torch.clamp(depth, min=1e-3, max=1e3)
        depth = torch.squeeze(depth).numpy()
        points = torch.squeeze(points).numpy()

        intrinsics = torch.from_numpy(trt_outputs[2].reshape((1,3,3)))
        intrinsics = postprocess_intrinsics(intrinsics, resize_factors)
        intrinsics = torch.squeeze(intrinsics).numpy()

        # Results - printed as before, and written to reports/bench/ so
        # compare.py can build the table without anyone retyping a number.
        bench.record('unidepth_v2', samples, encoder=encoder, warmup=warmup, precision=precision,
                     profile='bench', input_h=input_h, input_w=input_w,
                     engine_path=engine_file_path, outputs={'depth': depth},
                     model_input=batch_images)
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