# Metric3D V2
- **[Metric3Dv2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation](https://arxiv.org/abs/2404.15506)**
- **[Metric3D official GitHub](https://github.com/YvanYin/Metric3D)**
- 2d image -> depth,

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd Metric3D_V2
git clone https://github.com/YvanYin/Metric3D
cd Metric3D

# Create a new conda environment with Python 3.11
conda create -n metric3d -y python=3.11

# Activate the created environment
conda activate metric3d

# Install the required Python packages
pip install -r requirements_v2.txt
pip install -r requirements_v1.txt

```

2. run the original pytorch model on test images.
```
python hubconf.py
```
# see normal_vis.png
# abs_rel_err: 0.05135032907128334

3. check pytorch model inference performance
```
cd ..
python infer.py
```
- encoder: small, fp32, ipnut size: 616 x 1064 input
- 100 iterations time: 20.0422 [sec]
- Average FPS: 4.99 [fps]
- Average inference time: 200.42 [msec]
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
- encoder: small, fp32, ipnut size: 616 x 1064 input
- 100 iterations time: 16.4291 [sec]
- Average FPS: 6.09 [fps]
- Average inference time: 164.29 [msec]

**[Back](../README.md)** 