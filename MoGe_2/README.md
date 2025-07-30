# MoGe-2
- **[MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details](https://arxiv.org/abs/2507.02546)**
- **[MoGe-2 official GitHub](https://github.com/microsoft/MoGe)**
- 2d image -> depth, point cloud, surface normal, fov

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd MoGe_2
git clone https://github.com/microsoft/MoGe.git
cd MoGe

# Create a new conda environment with Python 3.11
conda create -n MoGe -y python=3.11

# Activate the created environment
conda activate MoGe

# Install the required Python packages
pip install -r requirements.txt 
pip install git+https://github.com/microsoft/MoGe.git
pip install pyglet==1.5.29
```

2. download pretrained checkpoints.
```
mkdir -p checkpoint/moge-2-vitl
wget https://huggingface.co/Ruicheng/moge-2-vitl/resolve/main/model.pt -P checkpoint/moge-2-vitl

mkdir -p checkpoint/moge-2-vitl-normal
wget https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt -P checkpoint/moge-2-vitl-normal

mkdir -p checkpoint/moge-2-vitb-normal
wget https://huggingface.co/Ruicheng/moge-2-vitb-normal/resolve/main/model.pt -P checkpoint/moge-2-vitb-normal

mkdir -p checkpoint/moge-2-vits-normal
wget https://huggingface.co/Ruicheng/moge-2-vits-normal/resolve/main/model.pt -P checkpoint/moge-2-vits-normal
```

3. run the original pytorch model on test images.
```
moge infer -i ../../data -o outputs --maps --glb --ply

```

4. check pytorch model inference performance
```
cd ..
python infer.py
```
- 518 x 518 input
- max : 2.4745407104492188 , min : 0.699651837348938
- 100 iterations time ((518, 518)): 14.0060 [sec]
- Average FPS: 7.14 [fps]
- Average inference time: 140.06 [msec]
--------------------------------------------------------------------

## How to Run (TensorRT)

1. generate onnx file

```
pip install onnx
pip install onnxsim
python onnx_export.py
// a file 'moge-2-vits-normal_dynamic_sim.onnx' will be generated in onnx directory.
```

2. build tensorrt model and run

```
conda activate trte
pip install matplotlib
python onnx2trt.py
// a file 'moge-2-vits-normal_fp16_dynamic_sim.engine' will be generated in engine directory.
```
- 518 x 518 input
- 100 iterations time ((518, 518)): 4.7010 [sec]
- Average FPS: 21.27 [fps]
- Average inference time: 47.01 [msec]
- max : 2.4711220264434814 , min : 0.7006123661994934

**[Back](../README.md)** 