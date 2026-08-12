# Metric3D V2
- **[Metric3Dv2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation](https://arxiv.org/abs/2404.15506)**
- **[Metric3D official GitHub](https://github.com/YvanYin/Metric3D)**
- 2d image -> depth, surface normal

> **Benchmark numbers below are historical and not comparable across models.**
> They were taken before the timing loop was unified, when warmup varied
> between 5 and 20, `depth_pro` timed its post-process inside the loop, and
> nothing recorded the GPU, driver or TensorRT version. Current results live
> in [reports/comparison.md](../reports/comparison.md), generated from
> `reports/bench/*.json` by `compare.py`. These lines are replaced as each
> model is re-measured.

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd Metric3D_V2
git clone https://github.com/YvanYin/Metric3D
cd Metric3D

# Create a new conda environment with Python 3.11
conda create -n metric3d -y python=3.11

# Activate the created environment
conda activate metric3d

# Install the required Python packages
pip install -r requirements_v2.txt
pip install -r requirements_v1.txt

```

2. run the original pytorch model on test images.
```
python hubconf.py
```
- see normal_vis.png
- abs_rel_err: 0.05135032907128334

3. check pytorch model inference performance
```
cd ..
python infer.py
```
- encoder: small, fp32, ipnut size: 616 x 1064 input
- 100 iterations time: 20.0422 [sec]
- Average FPS: 4.99 [fps]
- Average inference time: 200.42 [msec]
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
- encoder: small, fp32, ipnut size: 616 x 1064 input
- 100 iterations time: 16.4291 [sec]
- Average FPS: 6.09 [fps]
- Average inference time: 164.29 [msec]

## Note — the output is canonical depth, not metric depth

Metric3D trains on every dataset re-projected onto a single virtual camera of
focal length 1000px. The network predicts in that camera's frame, and upstream
converts back at the very end:

```python
canonical_to_real_scale = real_focal_length * scale / 1000.0
pred_depth = pred_depth * canonical_to_real_scale   # metres from here on
```

**The scripts here stop before that line**, because `real_focal_length` is not
known for an arbitrary image. `infer.py` keeps the block behind `if 0:` with
four candidate values that were tried and abandoned. Supply your own focal
length (EXIF, calibration, or another model's estimate) if you need metres.

Measured: the keep-ratio resize factor swings 4.4x across input sizes
(0.2716 → 1.1892) while the depth stays within scale 1.00–1.12. That
invariance is the signature of an output not yet tied to a real camera.

Practical effect — do not put this model in the same "metric" column as
`depth_pro` or `unidepth_v2` in a comparison table. See
[docs/model_contracts.md](../docs/model_contracts.md) D12 and §5.7.

Also note this folder uses **616×1064**, upstream's own size for the ViT
models, not the repo-wide 518 comparison size.

## License

| Item | License |
| :--- | :--- |
| Upstream code ([YvanYin/Metric3D](https://github.com/YvanYin/Metric3D)) | BSD-2-Clause |
| Conversion scripts in this folder | MIT (this repository) |

Upstream licences verified 2026-07-31. This table is a convenience summary — the upstream LICENSE file is authoritative.

**[Back](../README.md)** 