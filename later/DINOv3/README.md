# DINOv3
- **[DINOv3](https://ai.meta.com/research/publications/dinov3/)**
- **[DINOv3 official GitHub](https://github.com/facebookresearch/dinov3)**
- 

## How to Run (Pytorch)

conda deactivate 
conda env remove -n wildgs -y 

1. set up a virtual environment.
```
cd DINOv3
git clone https://github.com/facebookresearch/dinov3.git
cd dinov3

# Create a new conda environment with Python 3.11
conda create -n dinov3 -y python=3.11

# Activate the created environment
conda activate dinov3 

# Install the required Python packages
pip install -r requirements.txt

```

2. download pretrained checkpoints.
```

```

3. run the original pytorch model on test images.
```

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