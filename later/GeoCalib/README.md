# GeoCalib
- **[GeoCalib: Learning Single-image Calibration with Geometric Optimization](https://arxiv.org/pdf/2409.06704)**
- **[GeoCalib official GitHub](https://github.com/cvg/GeoCalib)**
- image ->  camera intrinsics, gravity direction

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd GeoCalib
git clone https://github.com/cvg/GeoCalib.git
cd GeoCalib

# Create a new conda environment with Python 3.11
conda create -n geocalib -y python=3.11

# Activate the created environment
conda activate geocalib 

# Install the required Python packages
python -m pip install -e .
pip install gradio
pip install spaces
pip install onnxsim
pip install onnx
pip install onnxscript
```

2. check pytorch model inference performance
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