# Environment setup and known pitfalls

## Measurement environment

The published TensorRT measurements were collected on an NVIDIA RTX 3080 with TensorRT 10.16.1.11 and a fixed 1800 MHz GPU clock. Accuracy and speed are reported separately because a model can be fast while producing relative or canonical depth rather than metres.

## Installation

Create the TensorRT environment with Python 3.11 and install `tensorrt-cu12<11`, `cuda-python<13`, ONNX, OpenCV, and Matplotlib. Keep upstream export environments separate from the TensorRT build environment. See the root README for the complete bootstrap command.

## Pitfall 1: TensorRT 11

TensorRT 11 uses a strongly typed builder API and no longer exposes the precision flags assumed by these scripts. Use TensorRT 10 until the build layer is migrated and revalidated.

## Pitfall 2: remote GPU jobs

Long jobs on the desktop GPU are launched through the repository's batch/scheduled-task helpers. Keep logs and generated JSON under `reports/`; do not treat an interrupted shell prompt as proof that the child process stopped.

## Pitfall 3: Windows Conda commands

On Windows, activate Conda from a batch-capable shell (`conda.bat`) or start a shell that has been initialized by Conda. A PowerShell command that returns immediately may have failed before the model script ran.

## Reproducibility checklist

1. Record GPU, TensorRT, CUDA, driver, model checkpoint, input size, and precision.
2. Build at least three repetitions for a configuration before changing the default.
3. Validate accuracy before accepting a faster engine.
4. Run `python tools/compare.py --check` before committing generated tables.
