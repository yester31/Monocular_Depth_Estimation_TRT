# VGGT
- **[VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651)**
- **[VGGT official GitHub](https://github.com/facebookresearch/vggt)**
- 2d image -> depth, (point cloud, extrinsic, intrinsic)

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd VGGT
git clone https://github.com/facebookresearch/vggt.git
cd vggt

# Create a new conda environment with Python 3.11
conda create -n vggt -y python=3.11

# Activate the created environment
conda activate vggt

# Install the required Python packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

2. run the original pytorch model on test images.
```
python demo_colmap.py --scene_dir=../../data --use_ba --max_query_pts=2048 --query_frame_num=5

```

4. check pytorch model inference performance
```
cd ..
python infer.py
```
- 518 x 518 input
- 50 iterations time ((518, 518)): 1052.2285 [sec]
- Average FPS: 0.05 [fps]
- Average inference time: 21044.57 [msec]
- max : 1.89297 , min : 0.69163
--------------------------------------------------------------------

## How to Run (TensorRT)

1. generate onnx file

```
pip install onnx
pip install onnxsim
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
- 100 iterations time: 13.9785 [sec]
- Average FPS: 7.15 [fps]
- Average inference time: 139.78 [msec]
- max : 1.90231 , min : 0.68998

**[Back](../README.md)** 