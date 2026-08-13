# by yhpark 2025-7-27
# generation pointcloud
import viser

from onnx2trt import *

# Start server
server = viser.ViserServer()
scene = server.scene

def update_point_cloud(points: np.ndarray, colors: np.ndarray = None):
    scene.add_point_cloud(
        "/dynamic_point_cloud",
        points=points,
        colors=colors,
        point_size=0.001  # 작게 설정 (기본값은 0.05)
    )

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results')
    os.makedirs(save_dir_path, exist_ok=True)
    input_h = 518 # 1036
    input_w = 518 # 1386

    # Input
    image_dir_name = 'video_frames'
    image_dir = os.path.join(CUR_DIR, '..',image_dir_name)
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]

    # List all files in the directory and filter only image files
    image_paths = [
        os.path.join(image_dir, fname) for fname in sorted(os.listdir(image_dir)) if os.path.splitext(fname)[1].lower() in valid_exts
    ]

    raw_image = cv2.imread(image_paths[0])
    h, w = raw_image.shape[:2]
    #h, w = 518,518
    print(f"[MDET] original image size : {raw_image.shape[:2]}")

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
    output_shape = (1, 518, 518)
    print(f'[MDET] trt output shape : {output_shape}')

    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine, output_shape, profile_idx=0)

        # Inference
        print('start')
        for idx, path in enumerate(image_paths):
            raw_image = cv2.imread(path)
            raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            raw_image = cv2.resize(raw_image_rgb, (input_w, input_h))
            input_image = preprocess_image(raw_image, input_h)  # Preprocess image
            batch_images = np.concatenate([input_image], axis=0)
            inputs[0].host = batch_images
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            torch.cuda.synchronize()
            # ===================================================================
            depth = torch.from_numpy(trt_outputs[0].reshape(output_shape))
            depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)
            depth = torch.clamp(depth, min=1e-3, max=1e3)
            depth = torch.squeeze(depth).numpy()
            # Generate mesh grid and calculate point cloud coordinates
            #focal_length = 3151.55 
            focal_length = 604
            #focal_length = 351.55 
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x = (x - w / 2) / focal_length
            y = (y - h / 2) / focal_length
            z = depth
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            colors = np.array(raw_image_rgb).reshape(-1, 3) / 255.0

            update_point_cloud(points, colors)
            #time.sleep(0.1)
            # print(f'count : {idx}')
    print('done!')
    # ===================================================================
    common.free_buffers(inputs, outputs, stream)


if __name__ == '__main__':
    main()