# NeuFlow
- **[NeuFlow-V2: Push High-Efficiency Optical Flow To the Limit](https://arxiv.org/abs/2408.10161)**
- **[NeuFlow official GitHub](https://github.com/neufieldrobotics/NeuFlow_v2)**
- 2d image -> optical flow

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd NeuFlow
git clone https://github.com/neufieldrobotics/NeuFlow_v2.git
cd NeuFlow_v2

# Create a new conda environment with Python 3.11
conda create -n neuflow -y python=3.11

# Activate the created environment
conda activate neuflow

# Install the required Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python
pip install huggingface-hub
```

2. run the original pytorch model on test images.
```
python infer.py
```

3. check pytorch model inference performance
```
cd ..
python ../gen_video2imgs.py
python infer.py
```
- input size: 288 x 512, iters: 
- 249 iterations time: 11.3614 [sec]
- Average FPS: 21.92 [fps]
- Average inference time: 45.63 [msec]
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
- input size: 288 x 512, iters: 
- 249 iterations time: 9.6196 [sec]
- Average FPS: 25.88 [fps]
- Average inference time: 38.63 [msec]

**[Back](../README.md)** 