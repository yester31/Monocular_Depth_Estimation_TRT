# Depth Anything V2
- **[Depth Anything V2](https://arxiv.org/abs/2406.09414)**
- **[Depth Anything V2 official GitHub](https://github.com/DepthAnything/Depth-Anything-V2)**
- 2d image -> depth

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd Depth_Anything_V2
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
conda create -n dav2 -y python=3.11
conda activate dav2
pip install -r requirements.txt
```

2. download pretrained checkpoints.
```
mkdir -p checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -P checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth -P checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -P checkpoints
```
3. run the original pytorch model on test images.
```
python run.py --encoder vits --img-path assets/examples --outdir results/vits
```

4. check pytorch model inference performance
```
cd ..
python infer.py
```
- 518 x 518 input
- see results/example_vits_Torch.png    
- 20 iterations time: 0.3925 [sec]
- Average FPS: 50.95 [fps]
- Average inference time: 19.63 [msec] 

## How to Run (TensorRT)

1. generate onnx file

```
pip install onnx
python onnx_export.py
// a file 'depth_anything_v2_vits.onnx' will be generated in onnx directory.
```

2. build tensorrt model and run

```
conda activate trte
pip install matplotlib
python onnx2trt.py
// a file 'depth_anything_v2_vits.engine' will be generated in engine directory.
```
- 518 x 518 input
- see results/example_vits_TRT.jpg  
- 100 iterations time: 0.8135 [sec]
- Average FPS: 122.92 [fps]
- Average inference time: 8.14 [msec]

**[Back](../README.md)** 