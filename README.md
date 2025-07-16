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

1. Depth Estimation  
    [Depth Pro](Depth_Pro/README.md)