# Monocular Depth Estimation Model to TensorRT
- Inference Optimization for Monocular Depth Estimation models using by TensorRT 
- Introducing various Monocular Depth Estimation models
- 2d to 3d 


## 1. Development Environment

- RTX3060 (notebook)
- WSL 
- Ubuntu 22.04.5 LTS
- cuda 12.9

```
conda create -n trte python=3.12 --yes 
conda activate trte
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install cuda-python
pip install tensorrt
pip install onnx
pip install opencv-python
pip install matplotlib
```

## 2. Monocular Depth Estimation Models 

### 2.1 Depth Anything V2
- **[Depth Anything V2 to TensorRT](Depth_Anything_V2/README.md)**   
### 2.2 Distill Any Depth
- **[Distill Any Depth to TensorRT](Distill_Any_Depth/README.md)**   
### 2.3 Depth Anything At Any Condition
- **[Depth Anything AC to TensorRT](Depth_Anything_AC/README.md)** 

### 2.4 Depth Pro
- **[Depth Pro to TensorRT](Depth_Pro/README.md)**   

### 2.5 Uni Depth V2
- **[Uni Depth V2 to TensorRT](Uni_Depth_V2/README.md)**  
### 2.6 Metric3D V2
- **[Metric3D V2 to TensorRT](Metric3D_V2/README.md)**  
### 2.7 UniK3D
- **[UniK3D to TensorRT](UniK3D/README.md)**  

### 2.8 MoGe-2
- **[MoGe-2 to TensorRT](MoGe_2/README.md)**  
### 2.9 VGGT
- **[VGGT to TensorRT](VGGT/README.md)**  
### 2.10 StreamVGGT
- **[StreamVGGT to TensorRT](StreamVGGT/README.md)**  


## 3. todo

### 3.9 Video Depth Anything
- **[Video Depth Anything to TensorRT](Video_Depth_Anything/README.md)** 
### 3.10 Flash Depth  
- **[Flash Depth to TensorRT](Flash_Depth/README.md)**   
### 3.11 Depth Crafter
- **[Depth Crafter to TensorRT](Depth_Crafter/README.md)**   
-------------
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)   
- [Flash Depth](https://github.com/Eyeline-Research/flashdepth)   
- [Depth Crafter](https://github.com/Tencent/DepthCrafter)   