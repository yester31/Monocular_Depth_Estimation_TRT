# VGGT
- **[Streaming 4D Visual Geometry Transformer](https://arxiv.org/abs/2507.11539)**
- **[StreamVGGT official GitHub](https://github.com/wzzheng/StreamVGGT)**
- 2d image -> depth, (point cloud, extrinsic, intrinsic, Camera pose)

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd StreamVGGT
git clone https://github.com/wzzheng/StreamVGGT.git
cd StreamVGGT

# Create a new conda environment with Python 3.11
conda create -n streamvggt -y python=3.11 cmake=3.14.0

# Activate the created environment
conda activate streamvggt

# Install the required Python packages
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
conda install 'llvm-openmp<16'
pip install onnx
pip install onnxsim
```

2. download pretrained checkpoints.
```
mkdir -p ckpt
wget https://huggingface.co/lch01/StreamVGGT/resolve/main/checkpoints.pth -P ckpt
```

3. check pytorch model inference performance
```
cd ..
python infer.py
```
- 518 x 518 input
- 10 iterations time : 394.1773 [sec]
- Average FPS: 0.03 [fps]
- Average Latency: 39417.73 [msec]
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
pip install matplotlib
python onnx2trt.py
// a file '.engine' will be generated in engine directory.
```
- 518 x 518 input
- 100 iterations time: 17.3173 [sec]
- Average FPS: 5.77 [fps]
- Average Latency: 173.17 [msec]

**[Back](../README.md)** 