# MEMFOF
- **[MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](https://arxiv.org/abs/2506.23151)**
- **[MEMFOF official GitHub](https://github.com/msu-video-group/memfof)**
- 2d image -> optical flow

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd MEMFOF
git clone https://github.com/msu-video-group/memfof.git
cd memfof

# Create a new conda environment with Python 3.11
conda create -n memfof -y python=3.11

# Activate the created environment
conda activate memfof

# Install the required Python packages
pip3 install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

2. run the original pytorch model on test images.
```
python demo.py
```

3. check pytorch model inference performance
```
cd ..
python ../gen_video2imgs.py
python infer.py
```
- input size: 288 x 512, iters: 8
- 249 iterations time: 41.1353 [sec]
- Average FPS: 6.05 [fps]
- Average inference time: 165.20 [msec]
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
- input size: 288 x 512, iters: 8
- 249 iterations time: 17.6442 [sec]
- Average FPS: 14.11 [fps]
- Average inference time: 70.86 [msec]

**[Back](../README.md)** 