# Video_Depth_Anything
- **[Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](https://arxiv.org/abs/2501.12375)**
- **[Video Depth Anything official GitHub](https://github.com/DepthAnything/Video-Depth-Anything)**
- video -> video depth

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd Video_Depth_Anything
git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything/

# Create a new conda environment with Python 3.11
conda create -n vda -y python=3.11

# Activate the created environment
conda activate vda 

# Install the required Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python
pip install tqdm
pip install einops
pip install easydict
pip install matplotlib
pip install imageio
pip install imageio-ffmpeg
pip install onnxsim
pip install onnx
pip install onnxscript
```

2. download pretrained checkpoints.
```
bash get_weights.sh
wget https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth -P metric_depth/checkpoints
```

3. run the original pytorch model on test images.
```
python run.py --input_video ./assets/example_videos/davis_rollercoaster.mp4 --output_dir ./outputs --encoder vitl

python run_streaming.py --input_video ./assets/example_videos/davis_rollercoaster.mp4 --output_dir ./outputs_streaming --encoder vitl

cd metric_depth python run.py --input_video ../assets/example_videos/davis_rollercoaster.mp4 --output_dir ../outputs_metric --encoder vitl
cd ..
```

4. check pytorch model inference performance
```
cd ..
python infer.py
```
--------------------------------------------------------------------

## How to Run (TensorRT)

1. generate onnx file
```
python onnx_export.py
// a file '.onnx' will be generated in onnx directory.
```

2. build tensorrt model and run
```
conda activate trte
python onnx2trt.py
// a file '.engine' will be generated in engine directory.
```

**[Back](../README.md)** 