# 환경 구성과 알려진 함정

README 의 설치 절에서 옮겨 왔다. 처음 한 번만 읽으면 되는 내용이다.

## 측정 환경

| | |
| --- | --- |
| 발표용 측정 | RTX 3080 10GB, TensorRT 10.16.1.11, 드라이버 591.86 |
| 개발 | RTX 3060 노트북 6GB — 엔진이 없어 코드·테스트만 |
| CUDA | 12.8 |

**GPU 클럭을 1800 MHz 로 고정하지 않고 빌드하지 않는다** (`nvidia-smi -lgc
1800,1800`, 해제는 `-rgc`). 이유와 근거 수치는 `PLAN.md` 실행 규칙 0.

## 설치

```bash
conda create -n trte python=3.11 --yes && conda activate trte
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "tensorrt-cu12<11" "cuda-python<13" onnx opencv-python matplotlib
```

모델마다 필요한 환경이 다르다. 각 `models/<name>/spec.json` 의 `env` 가 어느
환경인지 말하고, `run.py` 가 그것을 읽어 실행한다.

## 함정 1 — TensorRT 11 로 올리면 전부 깨진다

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

## 함정 2 — 원격 GPU 작업은 `schtasks` 로 띄운다

SSH 명령이 잘려도 데스크탑의 빌드 프로세스는 죽지 않는다. 잘린 작업이 쌓여
카드를 채우면 다음 측정이 오염된다. 실제로 P6 스윕 중 배치가 네 번 잘렸고,
GPU 가 10,042 MiB 로 찬 상태에서 돌린 빌드가 65분 동안 끝나지 않았다.

`schtasks` 로 띄우면 `schtasks /end` 로 확실히 끝낼 수 있고, 측정 전에
`nvidia-smi` 로 카드가 비었는지 확인할 수 있다.

## 함정 3 — Windows 에서 `conda` 는 배치 파일이다

`.bat` 안에서 `call` 없이 `conda run ...` 을 부르면 제어가 넘어가고 돌아오지
않는다. 배치가 첫 명령만 실행하고 조용히 멈춘 것처럼 보인다.

```bat
call conda run --no-capture-output -p %CONDA% python tools/tune_build.py ...
```
