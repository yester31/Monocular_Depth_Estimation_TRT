# Monocular_Depth_Estimation_TRT

## 0. Development Environment

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

[Depth Pro](Depth_Pro/README.md)   
[Depth Anything V2](Depth_Anything_V2/README.md)   
[Distill Any Depth](Distill_Any_Depth/README.md)   

- todo
   - [Depth Anything At Any Condition](DepthAnythingAC/README.md)   
   - [Video Depth Anything](Video_Depth_Anything/README.md)   
   - [Flash Depth](Flash_Depth/README.md)