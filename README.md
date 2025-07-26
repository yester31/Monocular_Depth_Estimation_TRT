# Monocular Depth Estimation Model to TensorRT
- Inference Optimization for Monocular Depth Estimation models using by TensorRT 
- Introducing various Monocular Depth Estimation models

## 0. todo
   - [Uni-Depth V2](Uni_Depth_V2/README.md)   
   - [Depth Anything At Any Condition](Depth_Anything_AC/README.md)   
   - [Zoe Depth](Zoe_Depth/README.md)   
   - [Metric3D V2](Metric3D_V2/README.md)   
   - [Video Depth Anything](Video_Depth_Anything/README.md)   
   - [Flash Depth](Flash_Depth/README.md)   
   - [Depth Crafter](Depth_Crafter/README.md)   

## 1. Development Environment

- RTX3060 (notebook)
- WSL 
- Ubuntu 22.04.5 LTS
- cuda 12.8


```
conda create -n trte python=3.12 --yes 
conda activate trte

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install cuda-python
pip install tensorrt
pip install onnx
pip install opencv-python

```

## 2. Monocular Depth Estimation Models 

### 2.1 Depth Pro
- **[Depth Pro to TensorRT](Depth_Pro/README.md)**   

### 2.2 Depth Anything V2
- **[Depth Anything V2](Depth_Anything_V2/README.md)**   

### 2.3 Distill Any Depth
- **[Distill Any Depth](Distill_Any_Depth/README.md)**   

### 2.4 MoGe-2
- **[MoGe-2](MoGe_2/README.md)**   


