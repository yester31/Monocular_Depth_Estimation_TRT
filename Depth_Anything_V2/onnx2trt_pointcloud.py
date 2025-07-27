# by yhpark 2025-7-27
# generation pointcloud
import open3d as o3d

from onnx2trt import *

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)
    input_h = 518 # 1036
    input_w = 518 # 1386

    # Input
    image_file_name = 'example.jpg'
    image_path = os.path.join(CUR_DIR, '..', 'data', image_file_name)
    raw_image = cv2.imread(image_path)
    raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    h, w = raw_image.shape[:2]
    print(f'[MDET] original shape : {raw_image.shape}')
    raw_image = cv2.resize(raw_image_rgb, (input_w, input_h))

    input_image = preprocess_image(raw_image, input_h)  # Preprocess image
    print(f'[MDET] after preprocess shape : {input_image.shape}')
    batch_images = np.concatenate([input_image], axis=0)

    # Model and engine paths
    precision = "fp16"  # Choose 'fp32' or 'fp16'
    encoder = 'vits'    # or 'vits', 'vitb', 'vitg'
    metric_model = True # True
    dataset = 'hypersim'# 'hypersim' for indoor model, 'vkitti' for outdoor model
    dynamo = True       # True or False
    onnx_sim = True     # True or False
    model_name = f"depth_anything_v2_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_metric_{dataset}" if metric_model else model_name
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

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)
        inputs[0].host = batch_images

        # Inference
        trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        # ===================================================================
        print('[MDET] Post process')
        depth = torch.from_numpy(trt_outputs[0].reshape(output_shape))
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        depth = torch.clamp(depth, min=1e-3, max=1e3)
        depth = torch.squeeze(depth).numpy()
        print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')

    common.free_buffers(inputs, outputs, stream)

    print('[MDET] Generate point cloud')
    # Generate mesh grid and calculate point cloud coordinates
    focal_length = 3365.20 # from depth pro
    # focal_length = 470.4 # from ori repo var
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x - w / 2) / focal_length
    y = (y - h / 2) / focal_length
    z = depth
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.array(raw_image_rgb).reshape(-1, 3) / 255.0

    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(save_dir_path, os.path.splitext(image_file_name)[0] + ".ply"), pcd)
    #o3d.visualization.draw_geometries([pcd])
    #o3d.visualization.draw([pcd])

    # ===================================================================


if __name__ == '__main__':
    main()