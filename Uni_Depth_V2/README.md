# UniDepthV2
- **[UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler](https://arxiv.org/abs/2502.20110)**
- **[UniDepthV2 official GitHub](https://github.com/lpiccinelli-eth/UniDepth)**
- 2d image -> depth, point cloud(xyz), Intrinsics, Camera rays, radius(distance from the origin)

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd Uni_Depth_V2
git clone https://github.com/lpiccinelli-eth/UniDepth.git
cd UniDepth

# Create a new conda environment with Python 3.11
conda create -n unidepthv2 -y python=3.11

# Activate the created environment
conda activate unidepthv2

# Install the required Python packages
pip install -r requirements.txt 
pip install -e .
```

2. checkpoints

No manual download step. Both `infer.py` and `onnx_export.py` pull the weights
from HuggingFace on first run and cache them:

| script | how |
| :--- | :--- |
| `infer.py` | `UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-<enc>14")` |
| `onnx_export.py` | `huggingface_hub.hf_hub_download(repo_id="lpiccinelli/unidepth-v2-<enc>14", filename="pytorch_model.bin")` |

Both resolve to the same repository, so the Torch reference and the exported
graph share weights. Set `HF_HOME` to control where the cache lives.

3. run the original pytorch model on test images.
```
python ./scripts/demo.py
```
# see demo/output.png

4. check pytorch model inference performance
```
cd ..
python infer.py
```
- encoder: vitb, input size: 518 x 518 
- 100 iterations time: 5.8027 [sec]
- Average FPS: 17.23 [fps]
- Average inference time: 58.03 [msec]
--------------------------------------------------------------------

## How to Run (TensorRT)

1. generate onnx file

```
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
- encoder: vitb, input size: 518 x 518 
- 100 iterations time: 2.7127 [sec]
- Average FPS: 36.86 [fps]
- Average inference time: 27.13 [msec]

## License

| Item | License |
| :--- | :--- |
| Upstream code ([lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth)) | **CC BY-NC 4.0 — non-commercial** |
| Conversion scripts in this folder | MIT (this repository) |

> **Note.** Non-commercial use only.

Upstream licences verified 2026-07-31. This table is a convenience summary — the upstream LICENSE file is authoritative.

**[Back](../README.md)** 