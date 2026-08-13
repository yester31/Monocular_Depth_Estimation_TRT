# by yhpark 2025-7-26
from onnx2trt import *

import threading
from collections import deque

class DepthWebcamStream:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.original_height = None
        self.original_width = None
        self.cmap = plt.get_cmap("turbo")
        
        self.CUR_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[MDET] using device: {DEVICE}")
        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        self.TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
        
        self.precision = "fp16"  # Choose 'fp32' or 'fp16'
        self.dynamo = True  # True or False
        self.model_name = "depth_pro_dynamo" if self.dynamo else "depth_pro"
        self.onnx_model_path = os.path.join(CUR_DIR, "onnx", f"{self.model_name}.onnx")
        self.engine_file_path = os.path.join(
            CUR_DIR, "engine", f"{self.model_name}_{self.precision}.engine"
        )
        os.makedirs(os.path.dirname(self.engine_file_path), exist_ok=True)
        
        self.transform = Compose([ToTensor(),Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
        self.engine = get_engine(self.onnx_model_path, self.engine_file_path, self.precision) 
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        
        self.running = False
        self.frame_queue = deque(maxlen=2) 
        self.result_queue = deque(maxlen=1) 
        self.lock = threading.Lock()
        
        self.last_webcam_time = time.time()
        self.webcam_fps = 0
        self.last_model_time = time.time()
        self.model_fps = 0
        self.first_step = True
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Can not open")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f"original shape : {self.original_height} x {self.original_width}")

        return True
    
    def process_frame(self, frame):
        try:
            start_model = time.time()

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = self.transform(image_rgb).unsqueeze(0)
            x = torch.nn.functional.interpolate(x, size=(1536, 1536), mode="bilinear", align_corners=False)

            # infer
            self.inputs[0].host = x.cpu().numpy()
            trt_outputs = common.do_inference(
                self.context,
                engine=self.engine,
                bindings=self.bindings,
                inputs=self.inputs,
                outputs=self.outputs,
                stream=self.stream,
            )
            canonical_inverse_depth = torch.from_numpy(trt_outputs[0].reshape((1, 1, 1536, 1536)))
            fov_deg = torch.from_numpy(trt_outputs[1])

            # post process
            f_px = (0.5 * self.original_width / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float))))
            inverse_depth = canonical_inverse_depth * (self.original_width / f_px)
            inverse_depth = torch.nn.functional.interpolate(
                inverse_depth,
                size=(self.original_height, self.original_width),
                mode="bilinear",
                align_corners=False,
            )
            inverse_depth = torch.squeeze(torch.clamp(inverse_depth, min=1e-4, max=1e4)).numpy()

            # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
            max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
            min_invdepth_vizu = max(1 / 250, inverse_depth.min())
            inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
                max_invdepth_vizu - min_invdepth_vizu
            )
            
            color_depth = (self.cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
            color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)

            self.model_fps = 1.0 / (time.time() - start_model + 1e-9)
            cv2.putText(color_depth_bgr, f"Model FPS: {self.model_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            if self.first_step:
                print(f_px) 
                self.first_step = False

            return color_depth_bgr
            
        except Exception as e:
            print(f"frame process error : {e}")
            return frame
    
    def model_worker(self):
        while self.running:
            frame = None
            with self.lock:
                if self.frame_queue:
                    frame = self.frame_queue.popleft()
            
            if frame is not None:
                depth_frame = self.process_frame(frame)
                
                # 결과 저장
                with self.lock:
                    self.result_queue.append((frame, depth_frame))
            else:
                time.sleep(0.001)
    
    def run_stream(self):
        if not self.start_camera():
            return
        
        self.running = True
        
        # start model thread
        model_thread = threading.Thread(target=self.model_worker)
        model_thread.start()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Something Wrong")
                    break
                
                now = time.time()
                self.webcam_fps = 1.0 / (now - self.last_webcam_time)
                self.last_webcam_time = now
                
                # insert frame to queue
                with self.lock:
                    self.frame_queue.append(frame.copy())
                
                # 처리된 결과가 있으면 표시
                result = None
                with self.lock:
                    if self.result_queue:
                        result = self.result_queue.popleft()
                
                if result is not None:
                    original_frame, depth_frame = result
                    cv2.putText(frame, f"Webcam FPS: {self.webcam_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    cv2.imshow('Original Frame', frame)
                    cv2.imshow('Depth Map', depth_frame)
                else:
                    cv2.putText(frame, f"Webcam FPS: {self.webcam_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    cv2.imshow('Original Frame', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("stoped stream")
        finally:
            self.running = False
            model_thread.join()  # terminate thread
            self.stop_camera()
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("stop camera")
        common.free_buffers(self.inputs, self.outputs, self.stream)


def main():
    print("Start Depth Pro WebCam Stream")
    
    # stream = DepthWebcamStream(camera_index=0)
    stream = DepthWebcamStream(camera_index='http://192.168.0.11:5000/video')
    
    stream.run_stream()

if __name__ == "__main__":
    main()
