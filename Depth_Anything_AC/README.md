# Depth Anything At Any Condition
- **[Depth Anything at Any Condition](https://arxiv.org/abs/2507.01634)**
- **[Depth Anything AC official GitHub](https://github.com/HVision-NKU/DepthAnythingAC)**
- 2d image -> depth

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd Depth_Anything_AC
git clone https://github.com/HVision-NKU/DepthAnythingAC.git
cd DepthAnythingAC

# Create a new conda environment with Python 3.11
conda create -n depth_anything_ac -y python=3.9

# Activate the created environment
conda activate depth_anything_ac

# Install the required Python packages
pip install -r requirements.txt 
```

2. download pretrained checkpoints.
```
# (Optional) Using huggingface mirrors
export HF_ENDPOINT=https://hf-mirror.com

# download DepthAnything-AC model from huggingface
huggingface-cli download --resume-download ghost233lism/DepthAnything-AC
```
# checkpoints/depth_anything_AC_vits.pth

3. run the original pytorch model on test images.
```
python tools/infer.py --input ../../data/example.jpg --output depth.png

```

4. check pytorch model inference performance
```
cd ..
python infer.py
```
- 518 x 518 input
- [MDET] 100 iterations time ((518, 518)): 1.7155 [sec]
- [MDET] Average FPS: 58.29 [fps]
- [MDET] Average inference time: 17.15 [msec]
- [MDET] max : 6.16797 , min : 0.00100

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
- [MDET] 100 iterations time: 0.7511 [sec]
- [MDET] Average FPS: 133.14 [fps]
- [MDET] Average inference time: 7.51 [msec]
- [MDET] max : 6.09375 , min : 0.00100

**[Back](../README.md)** 