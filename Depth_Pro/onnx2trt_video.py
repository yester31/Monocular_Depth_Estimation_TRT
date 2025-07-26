# by yhpark 2025-7-26
from onnx2trt import *

def main():

    # Input video
    video_file_name = 'video2.mp4'
    video_file_path = os.path.join(CUR_DIR, '..', 'data', video_file_name)
    # Output video
    output_video_file_path = os.path.join(CUR_DIR, 'results', f'{os.path.splitext(video_file_name)[0]}_trt.mp4')

    cap = cv2.VideoCapture(video_file_path)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"original shape : {original_width} x {original_height}")
    print(f"original fps : {original_fps}")
    print(f"original total_frames : {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file_path, fourcc, original_fps, (original_width, original_height))

    # Model and engine paths
    precision = "fp16"  # 'fp32' or 'fp16'
    dynamo = True # True or False
    model_name = "depth_pro_dynamo" if dynamo else "depth_pro"
    onnx_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')
    engine_file_path = os.path.join(CUR_DIR, 'engine', f'{model_name}_{precision}.engine')
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)
    transform = Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
    
    count = 0
    dur_time = 0
    # Load or build the TensorRT engine and do inference
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:
                
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            begin = time.time()
            # pre-proc
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = transform(image_rgb).unsqueeze(0)
            x = torch.nn.functional.interpolate(x, size=(1536, 1536), mode="bilinear", align_corners=False)

            # infer
            inputs[0].host = x.cpu().numpy()
            trt_outputs = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            canonical_inverse_depth = torch.from_numpy(trt_outputs[0].reshape((1, 1, 1536, 1536)))
            fov_deg = torch.from_numpy(trt_outputs[1])

            # post process
            f_px = 0.5 * original_width / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
            inverse_depth = canonical_inverse_depth * (original_width / f_px)
            inverse_depth = torch.nn.functional.interpolate(inverse_depth, size=(original_height, original_width), mode="bilinear", align_corners=False)
            depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
            depth = torch.squeeze(depth).numpy()
            inverse_depth = 1 / depth
            #inverse_depth = torch.squeeze(torch.clamp(inverse_depth, min=1e-4, max=1e4)).numpy()
            dur_time += time.time() - begin

            # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
            max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
            min_invdepth_vizu = max(1 / 250, inverse_depth.min())
            inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)

            cmap = plt.get_cmap("turbo")
            color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
            color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)

            count += 1
            fps = count / (dur_time + 1e-6)
            cv2.putText(color_depth_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(color_depth_bgr)

            #cv2.imshow('frame', color_depth_bgr)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Results
    print(f'[MDET] {count} iterations time ({x.shape}): {dur_time:.4f} [sec]')
    avg_time = dur_time / count
    print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
    print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')
    print(f'[MDET] focal_length : {f_px}') 
    common.free_buffers(inputs, outputs, stream)


if __name__ == '__main__':
    main()
