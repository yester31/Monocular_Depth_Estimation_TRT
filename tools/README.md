# Tools

The root contains only the model runner and the demo. All other executable utilities live here.

## Result and validation tools

| Tool | Purpose |
| --- | --- |
| `compare.py` | Generate and validate the speed comparison table from benchmark JSON |
| `evaluate_gt.py` | Score model outputs against the configured ground-truth manifest |
| `verify_accuracy.py` | Compare an engine with its ONNX reference and FP32 output |
| `models.py` | Report registered, built, measured, and stale models |
| `package_artifacts.py` | Assemble reproducible deployment artifacts |

## Investigation tools

| Tool | Purpose |
| --- | --- |
| `profile_model.py` | Attribute runtime to preprocessing, inference, and postprocessing layers |
| `tune_build.py` | Build repeated TensorRT variants while sweeping builder settings |
| `size_sweep.py` | Measure the accuracy impact of input resolution |
| `make_eval_manifest.py` | Create an explicit evaluation subset |
| `prepare_eval_inputs.py` | Regenerate evaluation crops for the supported aspect ratios |
| `sync_desktop.sh` | Synchronize source and reports with the remote GPU desktop |

## Convention checks

`check_diode_convention.py` and `check_pointmap.py` validate coordinate, scale, and point-map conventions. Run them after changing preprocessing or postprocessing.

## Retired tools

`tools/retired/` contains drivers for questions that have already been answered. They are retained for provenance and are not part of the normal build path. In particular, old video-frame helpers expect a user-provided local `video/video2.mp4`; that input is intentionally not tracked.

## Before a release

```bash
python tools/models.py --stale
python tools/compare.py --check
python -m pytest tests -q
git diff --check
```
