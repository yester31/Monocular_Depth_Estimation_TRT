# by yhpark 2025-7-30
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

from unidepth.models.unidepthv2.unidepthv2 import get_paddings, get_resize_factor


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
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    h, w = raw_image.shape[:2]
    print(f'[MDET] original shape : {raw_image.shape}')

    rgb_origin = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    #### ajust input size to fit pretrained model
    # keep ratio resize
    input_sizes = (616, 1064) # for vit model
    # input_sizes = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_sizes[0] / h, input_sizes[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_sizes[0] - h
    pad_w = input_sizes[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    #### normalize
    # Deliberately none here. Metric3DExportModel.forward() (upstream
    # onnx/metric3d_onnx_export.py) calls normalize_image() as its first step,
    # so mean/std are baked into the ONNX graph and it expects raw 0-255 input.
    # Normalising here as well would apply the statistics twice.
    x = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32)

    print(f'[MDET] after preprocess shape : {x.shape}')
    batch_images = np.concatenate([x], axis=0)

    # Model and engine paths (ff)
    precision = "fp32"  # 'fp32'
    encoder = 'vitl'    # 'vits' or vitl or vitg
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
    dur_time = 0
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision, dynamic_input_shapes, obey_precision_constraints=True) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
        inputs[0].host = batch_images
        
        #context.set_input_shape('input', batch_images.shape)

        # Warm-up      
        for _ in range(10):  
            common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        # Inference loop
        for _ in range(iteration):
            begin = time.time()
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

        # ===================================================================
        print('[MDET] Post process')
        pred_depth = torch.from_numpy(trt_outputs[0].reshape(output_shape)).squeeze()
        
        # Results
        print(f'[MDET] {iteration} iterations time: {dur_time:.4f} [sec]')
        avg_time = dur_time / iteration
        print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
        print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')
    # ===================================================================
    print('[MDET] Post process')
    # Metric Depth Estimation
    # un pad
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
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