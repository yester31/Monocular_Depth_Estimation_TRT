# UniK3D
- **[UniK3D: Universal Camera Monocular 3D Estimation](https://arxiv.org/abs/2503.16591)**
- **[UniK3D official GitHub](https://github.com/lpiccinelli-eth/unik3d)**
- 2d image -> depth, point cloud

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd UniK3D
git clone https://github.com/lpiccinelli-eth/UniK3D.git
cd UniK3D

# Create a new conda environment with Python 3.11
conda create -n unik3d -y python=3.11

# Activate the created environment
conda activate unik3d

# Install the required Python packages
pip install -r requirements.txt

```

2. download pretrained checkpoints.
```
mkdir -p checkpoints/vits
wget https://huggingface.co/lpiccinelli/unik3d-vits/resolve/main/model.safetensors -P checkpoints/vits

mkdir -p checkpoints/vitb
wget https://huggingface.co/lpiccinelli/unik3d-vitb/resolve/main/model.safetensors -P checkpoints/vitb

mkdir -p checkpoints/vitl
wget https://huggingface.co/lpiccinelli/unik3d-vitl/resolve/main/model.safetensors -P checkpoints/vitl

```
3. run the original pytorch model on test images.
```
python ./scripts/demo.py
python ./scripts/infer.py --input ./../../data/example.jpg --output results --config-file configs/eval/vitl.json --save --save-ply
```

4. check pytorch model inference performance
```
cd ..
python infer.py
```

- encoder: vitb, fp16, ipnut size: 518 x 518 input
- 100 iterations time ((518, 518)): 6.3899 [sec]
- Average FPS: 15.65 [fps]
- Average inference time: 63.90 [msec]
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
pip install cuda-python
pip install tensorrt
pip install matplotlib
python onnx2trt.py
// a file '.engine' will be generated in engine directory.
```
- encoder: vitb, fp16, ipnut size: 518 x 518 input
- 100 iterations time: 3.3076 [sec]
- Average FPS: 30.23 [fps]
- Average inference time: 33.08 [msec]

**[Back](../README.md)** 