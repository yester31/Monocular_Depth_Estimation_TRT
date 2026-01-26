# Monocular Depth Estimation Model to TensorRT

## Project Overview

This project aims to optimize the inference performance of various monocular depth estimation models using NVIDIA's TensorRT. It provides a pipeline to convert pre-trained PyTorch models into ONNX format and then into TensorRT engines, allowing for a comparative analysis of inference speeds.

- **Key Features:**
    - Introduction to various monocular depth estimation models and a TensorRT conversion pipeline.
    - Performance comparison (FPS, inference time) between the original PyTorch models and the TensorRT-optimized models.
    - Generation of 3D depth information and point clouds from 2D images.

## 1. Development Environment

- **Hardware:** NVIDIA RTX3060 (notebook)
- **OS:** Windows Subsystem for Linux (WSL)
- **Linux Distribution:** Ubuntu 22.04.5 LTS
- **CUDA Version:** 12.8

```bash
# Create and activate a Conda virtual environment
conda create -n trte python=3.11 --yes
conda activate trte

# Install the required libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install tensorrt-cu12
pip install onnx
pip install opencv-python
pip install matplotlib
```

## 2. Supported Models

Each model directory contains a `README.md` file with detailed instructions.

| Model Name | Link to TensorRT Conversion | Main Outputs |
| :--- | :--- | :--- |
| **Depth Anything V2** | [TensorRT Conversion](Depth_Anything_V2/README.md) | Depth |
| **Distill Any Depth** | [TensorRT Conversion](Distill_Any_Depth/README.md) | Depth |
| **Depth Anything AC** | [TensorRT Conversion](Depth_Anything_AC/README.md) | Depth |
| **Depth Pro** | [TensorRT Conversion](Depth_Pro/README.md) | Depth |
| **Uni Depth V2** | [TensorRT Conversion](Uni_Depth_V2/README.md) | Depth |
| **Metric3D V2** | [TensorRT Conversion](Metric3D_V2/README.md) | Depth |
| **UniK3D** | [TensorRT Conversion](UniK3D/README.md) | Depth |
| **MoGe-2** | [TensorRT Conversion](MoGe_2/README.md) | Depth |
| **VGGT** | [TensorRT Conversion](VGGT/README.md) | Depth |
| **StreamVGGT** | [TensorRT Conversion](StreamVGGT/README.md) | Depth |
| **Depth Anything V3** | [TensorRT Conversion](Depth_Anything_V3/README.md) | Depth |

---

## 3. To-Do List for Project Improvement

- [ ] **Unified Inference Script:** Create a single inference script that accepts the model name as an argument to improve user experience.
- [ ] **Summarize Performance Analysis:** Add a table to the main `README.md` that summarizes the performance of all models (including input resolution, precision, and hardware details) for easy comparison.
- [ ] **Docker Support:** Add a `Dockerfile` to facilitate the environment setup and ensure reproducibility.
---