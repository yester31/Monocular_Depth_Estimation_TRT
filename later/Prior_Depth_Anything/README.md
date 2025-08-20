# Prior Depth Anything
- **[Depth Anything with Any Prior](https://arxiv.org/abs/2505.10565)**
- **[Prior Depth Anything official GitHub](https://github.com/SpatialVision/Prior-Depth-Anything)**
- 2d image -> depth, (point cloud, extrinsic, intrinsic, Camera pose)

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd Prior_Depth_Anything
git clone https://github.com/SpatialVision/Prior-Depth-Anything.git
cd Prior-Depth-Anything

# Create a new conda environment with Python 3.11
conda create -n prior_depth_anything -y python=3.11

# Activate the created environment
conda activate prior_depth_anything

# Install the required Python packages
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install onnx
pip install onnxsim
```

2. run the original pytorch model on test images.
```
# We sample on Ground-Truth depth map as prior.
priorda test --image_path assets/sample-1/rgb.jpg --prior_path assets/sample-1/gt_depth.png --pattern downscale_32 --visualize 1 
```

3. check pytorch model inference performance
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
python onnx_export.py
// a file '.onnx' will be generated in onnx directory.
```

2. build tensorrt model and run

```
conda activate trte
python onnx2trt.py
// a file '.engine' will be generated in engine directory.
```
- 518 x 518 input
- 100 iterations time: 13.9785 [sec]
- Average FPS: 7.15 [fps]
- Average inference time: 139.78 [msec]
- max : 1.90231 , min : 0.68998

**[Back](../README.md)** 