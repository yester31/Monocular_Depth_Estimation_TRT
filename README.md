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

| Model Name | Link to TensorRT Conversion | Main Outputs | Upstream License |
| :--- | :--- | :--- | :--- |
| **Depth Anything V2** | [TensorRT Conversion](Depth_Anything_V2/README.md) | Depth | Apache-2.0 (code) / **CC BY-NC 4.0** (Base, Large weights) |
| **Distill Any Depth** | [TensorRT Conversion](Distill_Any_Depth/README.md) | Depth | MIT |
| **Depth Anything AC** | [TensorRT Conversion](Depth_Anything_AC/README.md) | Depth | **No license file** |
| **Depth Pro** | [TensorRT Conversion](Depth_Pro/README.md) | Depth | Apple Sample Code License |
| **Uni Depth V2** | [TensorRT Conversion](Uni_Depth_V2/README.md) | Depth | **CC BY-NC 4.0** |
| **Metric3D V2** | [TensorRT Conversion](Metric3D_V2/README.md) | Depth | BSD-2-Clause |
| **UniK3D** | [TensorRT Conversion](UniK3D/README.md) | Depth | **CC BY-NC-SA 4.0** |
| **MoGe-2** | [TensorRT Conversion](MoGe_2/README.md) | Depth | MIT |
| **VGGT** | [TensorRT Conversion](VGGT/README.md) | Depth | VGGT License (Meta custom) |
| **StreamVGGT** | [TensorRT Conversion](StreamVGGT/README.md) | Depth | **CC BY-NC-SA 4.0** |
| **Depth Anything V3** | [TensorRT Conversion](Depth_Anything_V3/README.md) | Depth | Apache-2.0 |
| **Metric Anything** | [TensorRT Conversion](Metric_Anything/README.md) | Depth | Apache-2.0 |

## 3. Licensing

The conversion scripts in this repository are MIT (see [LICENSE](LICENSE)). **The upstream models
are not.** Each model's `README.md` carries a `## License` table covering both the upstream code and
its checkpoints, which do not always match — Depth Anything V2 is Apache-2.0 but its Base and Large
weights are CC BY-NC 4.0.

Before using any model commercially, check that model's table. In particular:

- **Non-commercial:** Uni Depth V2, UniK3D, StreamVGGT, and the Depth Anything V2 Base/Large weights.
- **No license file at all:** Depth Anything AC — no usage rights are granted by default.
- **Custom licenses:** Depth Pro (Apple Sample Code License), VGGT (Meta's own license with an
  Acceptable Use Policy).

Upstream licences verified 2026-07-31 against the GitHub and HuggingFace APIs. The upstream LICENSE
file is always authoritative.

---

