# UniDepthV2
- **[UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler](https://arxiv.org/abs/2502.20110)**
- **[UniDepthV2 official GitHub](https://github.com/lpiccinelli-eth/UniDepth)**
- 2d image -> depth, point cloud(xyz), Camera rays, radius(distance from the origin), Intrinsics

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd Uni_Depth_V2
git clone https://github.com/lpiccinelli-eth/UniDepth.git
cd UniDepth

# Create a new conda environment with Python 3.11
conda create -n unidepthv2 -y python=3.11

# Activate the created environment
conda activate unidepthv2

# Install the required Python packages
pip install -r requirements.txt 
pip install -e .
```

2. run the original pytorch model on test images.
```
python ./scripts/demo.py
```
# see demo/output.png

3. check pytorch model inference performance
```
cd ..
python infer.py
```
- 518 x 518 input
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


**[Back](../README.md)** 