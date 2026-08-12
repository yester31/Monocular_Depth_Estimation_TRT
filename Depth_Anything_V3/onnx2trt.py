# by yhpark 2026-1-26
# Depth Anything V3 TensorRT model generation
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

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


    

def preprocess_image(raw_image):

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    image = image / 255.0
    
    # NormalizeImage
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # PrepareForNet
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image).astype(np.float32)

    # [C, H, W] -> [1, C, H, W]
    image = np.expand_dims(image, axis=0)

    return image

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)

    input_h = 518 # 1036
    input_w = 518 # 1386

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_img = cv2.imread(image_path)
    h, w = raw_img.shape[:2]
    print(f'[MDET] original shape : {raw_img.shape}')
    raw_img = cv2.resize(raw_img, (input_w, input_h))

    input_image = preprocess_image(raw_img)  # Preprocess image
    print(f'[MDET] after preprocess shape : {input_image.shape}')
    batch_images = np.concatenate([input_image], axis=0)

    # Model and engine paths
    precision = "fp16"  # 'fp32' or 'fp16'
    metric_model = True # True or False
    onnx_sim = True     # True or False
    model_name = f"DA3METRIC-LARGE_{input_h}x{input_w}"
    model_name = f"{model_name}_sim" if onnx_sim else model_name
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # input & output shapes 
    input_shape = (batch_images.shape)
    output_shape = (1, batch_images.shape[2], batch_images.shape[3])
    print(f'[MDET] trt input shape : {input_shape}')
    print(f'[MDET] trt output shape : {output_shape}')

    dynamic_input_shapes = None

    iteration = 100
    dur_time = 0
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision, dynamic_input_shapes) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
        inputs[0].host = batch_images
        
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
        depth = torch.from_numpy(trt_outputs[0].reshape(output_shape))
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        depth = torch.clamp(depth, min=1e-3, max=1e3)
        depth = torch.squeeze(depth).numpy()

        # Results
        print(f'[MDET] {iteration} iterations time: {dur_time:.4f} [sec]')
        avg_time = dur_time / iteration
        print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
        print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')
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