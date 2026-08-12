# by yhpark 2025-7-16
# Depth Pro TensorRT model generation
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import tensorrt as trt
import common
from common import *

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
import open3d as o3d

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO

    
def main():

    iteration = 20
    img_size = 1536
    interpolation_mode = "bilinear"

    # Input
    f_px0 = None
    image_file_name = 'example.jpg'
    # image_file_name = '1.png'
    # image_file_name = 'panda0.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    h, w = raw_image.shape[:2]
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
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
    model_name = "depth_pro_dynamo" if dynamo else "depth_pro"
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    # Output shapes 
    output_shapes = (1, 1, img_size, img_size)
    print(f'[MDET] trt output shape : {output_shapes}')

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision, workspace_gib=3) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        inputs[0].host = x

        # Warm-up      
        for _ in range(5):  
            common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        # Inference loop
        dur_time = 0
        for _ in range(iteration):
            begin = time.time()
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            
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
            
            torch.cuda.synchronize()
            dur_time += time.time() - begin

        # Results
        print(f'[MDET] {iteration} iterations time ({x.shape}): {dur_time:.4f} [sec]')
        avg_time = dur_time / iteration
        print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
        print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')

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
    # Generate mesh grid and calculate point cloud coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x - w / 2) / f_px
    y = (y - h / 2) / f_px
    z = depth
    points = np.stack((x * z, y * z, z), axis=-1).reshape(-1, 3)
    colors = np.array(image_rgb).reshape(-1, 3) / 255.0

    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + ".ply"), pcd)
    o3d.visualization.draw_geometries([pcd])
    #o3d.visualization.draw([pcd])
    
if __name__ == '__main__':
    main()
