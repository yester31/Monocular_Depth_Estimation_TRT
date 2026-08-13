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
pip install "tensorrt-cu12<11"
pip install "cuda-python<13"
pip install onnx
pip install opencv-python
pip install matplotlib
```

> **`tensorrt-cu12` must be pinned below 11.** TensorRT 11 is strongly typed:
> it removed `BuilderFlag.FP16`, `INT8`, `BF16` and `OBEY_PRECISION_CONSTRAINTS`,
> along with `builder.platform_has_fast_fp16`. Precision now comes from the
> types in the ONNX graph rather than from a builder flag. Every script here
> selects precision with `precision = "fp16"`, so on TensorRT 11 the build
> fails with:
>
> ```
> AttributeError: type object 'BuilderFlag' has no attribute 'FP16'
> ```
>
> Verified on 11.2.1.2; 10.16.1.11 works. Porting to the strongly-typed API
> means baking the precision into the ONNX at export time and re-establishing
> every accuracy baseline, so it is tracked as its own task rather than done
> in passing.
>
> **`cuda-python` must be pinned below 13.** `common_runtime.py` needs the CUDA
> driver/runtime bindings. Version 13 removed the top-level
> `from cuda import cuda, cudart`, and its `cuda-bindings` wheel does not always
> ship `cuda.bindings.driver` / `.runtime` either, so a plain
> `pip install cuda-python` leaves the runtime unusable.
>
> Install **`tensorrt-cu12`**, not the `tensorrt` metapackage — that one
> currently pulls a CUDA 13 build which fails on drivers below 580 with
> `createInferBuilder: Error Code 6: CUDA initialization failure with error: 35`.
> Check your driver with `nvidia-smi`.
>
> **On a Korean or other non-UTF-8 Windows console**, set `PYTHONUTF8=1` before
> running `onnx_export.py`. `torch.onnx` prints a ✅ that cp949 cannot encode,
> and the export dies with `UnicodeEncodeError` after doing all the work.

`moge_2` and `metric_anything` need two more packages here, because their
post-process calls MoGe's `recover_focal_shift` rather than reading it off the
engine (see `docs/model_contracts.md` D15):

```bash
pip install trimesh
pip install "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"
```

> **`utils3d` must be that commit, not the PyPI release.** Both upstreams pin
> it exactly. The PyPI package of the same name has a different API and fails
> only in post-process, after the engine has been built and benchmarked:
>
> ```
> AttributeError: module 'utils3d' has no attribute 'torch'
> ```

## 2. Layout — moved 2026-08-13

Every model now lives under `models/` with a lowercase directory name matching
the key used in reports and result files. Before this, the same model went by
three spellings — the directory `Uni_Depth_V2`, the key `unidepth_v2`, and
upstream's `UniDepthV2` — and code had to carry a mapping between them.

| was | now |
| :--- | :--- |
| `Depth_Anything_V2/` | `models/depth_anything_v2/` |
| `Depth_Anything_AC/` | `models/depth_anything_ac/` |
| `Depth_Anything_V3/` | `models/depth_anything_v3/` |
| `Distill_Any_Depth/` | `models/distill_any_depth/` |
| `Depth_Pro/` | `models/depth_pro/` |
| `Metric3D_V2/` | `models/metric3d_v2/` |
| `Metric_Anything/` | `models/metric_anything/` |
| `MoGe_2/` | `models/moge_2/` |
| `StreamVGGT/` | `models/streamvggt/` |
| `UniK3D/` | `models/unik3d/` |
| **`Uni_Depth_V2/`** | **`models/unidepth_v2/`** — note the name change |
| `VGGT/` | `models/vggt/` |

Commands gain one path component:

```bash
cd models/depth_anything_v2 && python onnx_export.py    # was: cd Depth_Anything_V2
```

Scripts no longer count `..` to find the repository root; they walk up to the
directory containing `core/`. That is why the move did not require touching
every path by hand, and why the next move will not either.

If you have upstream clones inside the old directories, move them with the
rest — the model scripts still expect them alongside, e.g.
`models/vggt/vggt/`.

## 3. Commands

Five scripts at the root. The first three read `spec.json` and `reports/` and
import nothing from a model directory, so they run on a laptop with no CUDA —
answering "what is in here" should not need twelve conda environments.

| | |
| :--- | :--- |
| `python models.py` | what exists, what is built, what is measured (`--stale` for the gaps) |
| `python compare.py` | regenerate `reports/comparison.md` and the tables in this file (`--check` to fail if stale) |
| `python package_artifacts.py` | assemble `artifacts/<model>/<profile>-<precision>/` with a manifest (`--verify`) |
| `python verify_accuracy.py` | each engine against its own ONNX at fp32 → `reports/accuracy.md` |
| `python run.py <stage> <model>` | run a model's script in the environment `spec.json` says it needs |

`run.py` exists because the environment depends on the stage, not the model:
exporting ONNX needs the upstream package, building an engine needs TensorRT,
and those two deliberately do not live in one environment.

```bash
python run.py export unidepth_v2     # ONNX, in the model's own env
python run.py build  unidepth_v2     # engine + measurement, in trte
python run.py build  --all           # all twelve, each in its own env
python run.py build  --all --dry-run # print the commands, run nothing
```

The model scripts take no arguments — encoder, resolution and precision are
constants edited in the source — so `run.py` has nothing to pass through and
does not pretend otherwise. `--dry-run` works anywhere, including where none of
the environments exist.

## 4. Supported Models

Each model directory contains a `README.md` file with detailed instructions.

Speeds are in [reports/comparison.md](reports/comparison.md), generated from
measurements rather than typed in. Input size is part of each row, because
three of these models do not run at 518 and attention cost grows faster than
pixel count — see that file before ranking anything.

**"Depth" is not one thing.** These models return different quantities, and a
few need a caveat that travels with the number:

| Model Name | Link to TensorRT Conversion | Input | Output | Upstream License |
| :--- | :--- | :--- | :--- | :--- |
| **Depth Anything V2** | [TensorRT Conversion](models/depth_anything_v2/README.md) | 518×518 | Metric (hypersim) by default; relative checkpoint also ships | Apache-2.0 (code) / **CC BY-NC 4.0** (Base, Large weights) |
| **Distill Any Depth** | [TensorRT Conversion](models/distill_any_depth/README.md) | 518×518 | Relative | MIT |
| **Depth Anything AC** | [TensorRT Conversion](models/depth_anything_ac/README.md) | 518×518 | Relative | **No license file** |
| **Depth Pro** | [TensorRT Conversion](models/depth_pro/README.md) | **1536×1536** | Metric + focal length | Apple Sample Code License |
| **Uni Depth V2** | [TensorRT Conversion](models/unidepth_v2/README.md) | 518×518 | Point map + intrinsics — **metric scale off ~3.1×, see D11** | **CC BY-NC 4.0** |
| **Metric3D V2** | [TensorRT Conversion](models/metric3d_v2/README.md) | **616×1064** | **Canonical depth, not metres — see D12** | BSD-2-Clause |
| **UniK3D** | [TensorRT Conversion](models/unik3d/README.md) | 518×518 | Point map + intrinsics — **metric scale 3.15× off, see D11** | **CC BY-NC-SA 4.0** |
| **MoGe-2** | [TensorRT Conversion](models/moge_2/README.md) | **388×518** | Point map + metric scale | MIT |
| **VGGT** | [TensorRT Conversion](models/vggt/README.md) | 518×518 | Geometry; scale unknown | VGGT License (Meta custom) |
| **StreamVGGT** | [TensorRT Conversion](models/streamvggt/README.md) | 518×518 | Geometry; scale unknown | **CC BY-NC-SA 4.0** |
| **Depth Anything V3** | [TensorRT Conversion](models/depth_anything_v3/README.md) | 518×518 | Relative + sky mask | Apache-2.0 |
| **Metric Anything** | [TensorRT Conversion](models/metric_anything/README.md) | **388×518** | Point map + metric scale | Apache-2.0 |

The D-numbers refer to [docs/model_contracts.md](docs/model_contracts.md),
which records what was measured and how.

**Not converted yet.** `later/` holds 16 upstream clones that were attempted
and set aside — none of them has an engine, a benchmark or a `spec.json`, and
nothing in this repository reads from that directory. They are kept because
the clone and its notes are the expensive part, not the conversion.

## 5. Performance

<!-- BENCH:BEGIN -->
Measured on NVIDIA GeForce RTX 3080, TensorRT 10.16.1.11, on `data/example.jpg`.
Generated by `compare.py` — do not edit between the markers.

**Input 518x518**

| model               | precision | mean ms | p50   | fps    | output                       |
| --- | --- | ---: | ---: | ---: | --- |
| `depth_anything_v2` | fp16      | 4.27    | 4.26  | 234.45 | metric (hypersim) by default |
| `distill_any_depth` | fp16      | 4.32    | 4.31  | 231.72 | relative                     |
| `depth_anything_ac` | fp16      | 4.36    | 4.36  | 229.14 | relative                     |
| `unidepth_v2`       | fp16      | 8.21    | 8.20  | 121.85 | point map + intrinsics       |
| `unik3d`            | fp16      | 8.44    | 8.45  | 118.47 | point map + intrinsics       |
| `depth_anything_v3` | fp16      | 19.66   | 19.66 | 50.85  | metric + sky mask            |
| `vggt`              | fp16      | 52.90   | 52.90 | 18.90  | geometry (scale unknown)     |
| `streamvggt`        | fp16      | 53.35   | 53.34 | 18.74  | geometry (scale unknown)     |

**Input 388x518**

| model             | precision | mean ms | p50   | fps   | output                   |
| --- | --- | ---: | ---: | ---: | --- |
| `moge_2`          | fp16      | 19.67   | 19.66 | 50.84 | point map + metric_scale |
| `metric_anything` | fp16      | 67.84   | 67.82 | 14.74 | point map + metric_scale |

**Input 1536x1536**  (only model at this size)

| model       | precision | mean ms | p50    | fps  | output |
| --- | --- | ---: | ---: | ---: | --- |
| `depth_pro` | fp16      | 241.24  | 241.29 | 4.15 | metric |

**Input 616x1064**  (only model at this size)

| model         | precision | mean ms | p50   | fps   | output                       |
| --- | --- | ---: | ---: | ---: | --- |
| `metric3d_v2` | fp32      | 62.37   | 62.38 | 16.03 | canonical (not metres - D12) |

Speeds do **not** compare across those groups: attention cost grows faster than pixel count, so a model at a smaller input is not therefore faster. Full table with p90/p99 and the per-model caveats: [reports/comparison.md](reports/comparison.md).
<!-- BENCH:END -->

## 6. Licensing

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

