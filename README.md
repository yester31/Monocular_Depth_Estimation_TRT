# Monocular Depth Estimation → TensorRT

단안 깊이 추정 모델 14개를 TensorRT 엔진으로 변환하고, **같은 조건에서 속도와
정확도를 비교한다.**

비교가 성립하려면 조건이 같아야 하는데 이 모델들은 조건이 같지 않다 — 입력
크기가 다르고(388×518부터 1536×1536까지), 출력의 의미가 다르고(미터 · 상대
깊이 · 정규화 좌표), 같은 "metric" 이라도 보정 방식이 다르다. 그래서 이
저장소는 **숫자보다 그 숫자가 무엇인지를 먼저 기록한다.**

| | |
| --- | --- |
| 측정 환경 | RTX 3080, TensorRT 10.16.1.11, GPU 클럭 1800 MHz 고정 |
| 속도 | [`reports/comparison.md`](reports/comparison.md) — `compare.py` 가 JSON 에서 생성 |
| 정답 데이터 정확도 | [`reports/gt.md`](reports/gt.md) — DIODE 실내 50장 |
| 엔진 대 ONNX | [`reports/accuracy.md`](reports/accuracy.md) |

---

## 무엇이 어디에 있나

| 경로 | 내용 |
| --- | --- |
| `models/<name>/` | 모델마다 `spec.json` · `onnx_export.py` · `onnx2trt.py` · `README.md` |
| `core/` | 공유 구현 — 전처리, 벤치마크, 정답 데이터 지표, 빌드 조건, 시각화 |
| `tools/` | **루트에 없는 실행 가능한 것 전부.** 무엇이 무엇이고 어떻게 쓰는지는 [`tools/README.md`](tools/README.md) |
| `tools/retired/` | 답이 나와서 다시 돌릴 일이 없는 실험 드라이버. 지우지 않고 보관 |
| `reports/` | **측정 결과의 단일 출처.** `.md` 는 전부 `.json` 에서 생성된다 |
| `tests/` | 회귀 테스트 |
| `docs/` | 아래 "문서" 절 |

---

## 시작하기

```bash
conda create -n trte python=3.11 --yes && conda activate trte
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "tensorrt-cu12<11" "cuda-python<13" onnx opencv-python matplotlib
```

> **`tensorrt-cu12` 는 11 미만으로 고정해야 한다.** TensorRT 11 은 strongly
> typed 로 바뀌면서 `BuilderFlag.FP16` · `INT8` · `BF16` 을 없앴다. 이
> 저장소의 스크립트는 전부 `precision = "fp16"` 으로 정밀도를 고르므로 11 에서는
> 빌드가 실패한다. 자세한 실패 메시지와 배경은
> [`docs/setup.md`](docs/setup.md).

모델마다 필요한 환경이 다르다. ONNX 를 내보내려면 업스트림 패키지가, 엔진을
빌드하려면 TensorRT 가 필요하고 **둘은 의도적으로 한 환경에 두지 않는다.**
`run.py` 가 `spec.json` 을 읽어 알맞은 환경에서 실행한다.

---

## 명령

루트에 있는 것은 둘이다 — **모델을 돌린다**와 **결과를 본다**.

```bash
python run.py export unidepth_v2      # ONNX, 모델 자기 환경에서
python run.py build  unidepth_v2      # 엔진 + 측정, trte 에서
python run.py build  --all            # 14개 전부, 각자의 환경에서
python run.py build  --all --dry-run  # 명령만 출력

python demo.py                        # 같은 입력, 모든 모델 출력을 같은 축에
python demo.py --synthetic            # GPU 없이 레이아웃만 확인
```

모델 스크립트는 인자를 받지 않는다. 인코더·해상도·정밀도는 소스에 상수로 있다.
`run.py` 가 있는 이유는 **환경이 모델이 아니라 단계에 달려 있기** 때문이다 —
ONNX 내보내기는 업스트림 패키지가, 엔진 빌드는 TensorRT 가 필요하고 둘은
의도적으로 한 환경에 없다.

그 밖의 모든 실행 가능한 것은 `tools/` 에 있고, 무엇이 무엇인지는
[`tools/README.md`](tools/README.md) 가 답한다. 자주 쓰는 것만 옮겨 적으면:

| | |
| --- | --- |
| `python tools/models.py` | 무엇이 있고 · 빌드됐고 · 측정됐나 (`--stale` 로 빈칸만) |
| `python tools/compare.py --check` | 발표 표가 JSON 과 어긋났는지. **커밋 전에 돌린다** |
| `python tools/evaluate_gt.py` | 정답 깊이 데이터로 채점 |
| `python tools/verify_accuracy.py` | 엔진을 자기 ONNX 와 fp32 에서 대조 |

앞의 둘은 `spec.json` 과 `reports/` 만 읽으므로 **CUDA 없는 노트북에서도 돈다** —
"여기 뭐가 있나" 에 답하는 데 conda 환경 14개가 필요해서는 안 되기 때문이다.

---

## 모델 15개

**"깊이" 는 한 가지가 아니다.** 아래 표의 `출력` 열을 보지 않고 속도만 비교하면
안 된다. D 번호는 [`docs/model_contracts.md`](docs/model_contracts.md) 의 기록을
가리킨다.

| 모델 | 변환 문서 | 입력 | 출력 | 업스트림 라이선스 |
| :--- | :--- | :--- | :--- | :--- |
| **Depth Anything V2** | [→](models/depth_anything_v2/README.md) | 518×518 | 기본 metric(hypersim), relative 체크포인트도 있음 | Apache-2.0 (코드) / **CC BY-NC 4.0** (Base·Large 가중치) |
| **Depth Anything V3** | [→](models/depth_anything_v3/README.md) | 518×518 | metric + 하늘 마스크 | Apache-2.0 |
| **Depth Anything AC** | [→](models/depth_anything_ac/README.md) | 518×518 | relative | **라이선스 파일 없음** |
| **Distill Any Depth** | [→](models/distill_any_depth/README.md) | 518×518 | relative | MIT |
| **ZipDepth** | [→](models/zipdepth/README.md) | **384×512** | affine-invariant 역깊이. **여기서 유일한 합성곱 모델**(6.1M) | 모델 README 참조 |
| **Depth Pro** | [→](models/depth_pro/README.md) | **1536×1536** | metric + 초점거리 | Apple Sample Code License |
| **Metric3D V2** | [→](models/metric3d_v2/README.md) | **616×1064** | **canonical depth, 미터 아님 — D12** | BSD-2-Clause |
| **Metric Anything** | [→](models/metric_anything/README.md) | **388×518** | point map + metric scale | Apache-2.0 |
| **MoGe-2** | [→](models/moge_2/README.md) | **388×518** | point map + metric scale | MIT |
| **Uni Depth V2** | [→](models/unidepth_v2/README.md) | **672×896** | point map + intrinsics | **CC BY-NC 4.0** |
| **UniK3D** | [→](models/unik3d/README.md) | **672×896** | point map + intrinsics | **CC BY-NC-SA 4.0** |
| **TR2M** | [→](models/tr2m/README.md) | **434×560** | metric. **입력이 둘 — 이미지 + 텍스트 프롬프트** | 업스트림 LICENSE 없음 / pos_embed.py 는 CC BY-NC-SA |
| **VGGT** | [→](models/vggt/README.md) | 518×518 | geometry, **전역 배율 없음**(정규화 좌표) | VGGT License (Meta 자체) |
| **StreamVGGT** | [→](models/streamvggt/README.md) | 518×518 | geometry, **전역 배율 없음** | **CC BY-NC-SA 4.0** |
| **HyDen** | [→](models/hyden/README.md) | 518×518 | relative. **방향 미측정** — 아직 벤치 없음 | FAIR Noncommercial Research |

### 인코더·체크포인트 선택

여기서 `사용 가능`은 **업스트림이 실제 가중치를 공개했고 이 저장소의 모델 로더가
그 구성을 알고 있다**는 뜻이다. 클래스 안에 이름만 정의돼 있거나 체크포인트가
`Coming soon`인 구성은 포함하지 않는다. 현재 발표 엔진과 정확도 표는 전부
`현재 사용` 열의 구성만 측정한 결과다. 다른 인코더로 바꾸면 새 ONNX·엔진을 만들고
속도와 정확도를 다시 검증해야 한다.

지금은 CLI 옵션이 없으므로 각 모델의 `onnx_export.py`와 `onnx2trt.py`에 있는
상수를 함께 바꿔야 한다. 한쪽만 바꾸면 다른 체크포인트의 엔진을 잘못 읽을 수 있다.

| 모델 | 현재 사용 | 사용 가능한 인코더·변형 | 설명 |
| :--- | :--- | :--- | :--- |
| **Depth Anything V2** | `vits` | `vits`, `vitb`, `vitl` | DINOv2 Small/Base/Large. `vitg` 코드 구성은 있지만 공식 Giant 체크포인트는 아직 `Coming soon`이므로 사용 가능 목록에서 제외 |
| **Depth Anything V3** | `DA3Metric-Large` | `DA3Metric-Large` | 이 저장소가 통합한 metric Large 체크포인트 한 종류 |
| **Depth Anything AC** | `vits` | `vits` | 로더에는 Base/Large 형상도 있으나 공개·확인된 AC 체크포인트는 Small뿐 |
| **Distill Any Depth** | `small` | `small`, `base`, `large` | 각각 Depth Anything 계열 ViT-S/ViT-B/ViT-L 체크포인트. 이름은 인코더명보다 업스트림 체크포인트 변형명에 가깝다 |
| **ZipDepth** | `base` | `base` | `base_npu`는 같은 6.1M 구조의 연산자 구현 차이이며 별도 인코더가 아님 |
| **Depth Pro** | 고정 | 선택 없음 | patch/image/FoV 경로가 모두 `dinov2l16_384`인 단일 공개 체크포인트 |
| **Metric3D V2** | `vits` | `vits`, `vitl`, `vitg2` | ViT-S는 RAFT 4회, ViT-L은 8회라 단순한 백본 크기 교체만은 아님. ConvNeXt-T/L은 별도 V1 계열 |
| **Metric Anything** | 고정 | 선택 없음 | `student_pointmap.pt` 단일 체크포인트 |
| **MoGe-2** | `vits` | `vits`, `vitb`, `vitl` | 현재 엔진은 normal head를 포함한 `vits-normal` 변형 |
| **UniDepth V2** | `vits` | `vits`, `vitb`, `vitl` | DINOv2 ViT-S/B/L 계열 |
| **UniK3D** | `vits` | `vits`, `vitb`, `vitl` | DINOv2 ViT-S/B/L 계열 |
| **TR2M** | 고정 조합 | 선택 없음 | Depth Anything ViT-S + DINOv2 ViT-L + CLIP ViT-L/14. 공개된 ScaleMap head가 이 조합 하나라 일부만 바꿀 수 없음 |
| **VGGT** | 고정 | 선택 없음 | VGGT-1B 단일 체크포인트; 별도 encoder 인자를 노출하지 않음 |
| **StreamVGGT** | 고정 | 선택 없음 | 공개 StreamVGGT 체크포인트 한 종류 |

`unidepth_v2` · `unik3d` 의 672×896 은 원래 518 이었다. 518 을 강제했을 때
metric 배율이 3.1배 어긋났는데, 모델 성질이 아니라 크기를 강제한 대가였다
([`docs/findings.md`](docs/findings.md) P3).

---

## 성능

<!-- BENCH:BEGIN -->
Measured on NVIDIA GeForce RTX 3080, TensorRT 10.16.1.11, on `data/example.jpg`.
Generated by `compare.py` — do not edit between the markers.

**Input 518x518**

| model               | precision | mean ms | p50   | fps    | output                       |
| --- | --- | ---: | ---: | ---: | --- |
| `depth_anything_v2` | fp16      | 4.31    | 4.31  | 232.11 | metric (hypersim) by default |
| `distill_any_depth` | fp16      | 4.32    | 4.32  | 231.47 | relative                     |
| `depth_anything_ac` | fp16      | 4.38    | 4.37  | 228.47 | relative                     |
| `depth_anything_v3` | fp16      | 19.90   | 19.89 | 50.26  | metric + sky mask            |
| `vggt`              | fp16      | 52.58   | 52.58 | 19.02  | geometry (scale unknown)     |
| `streamvggt`        | fp16      | 52.83   | 52.82 | 18.93  | geometry (scale unknown)     |

**Input 388x518**

| model             | precision | mean ms | p50   | fps   | output                   |
| --- | --- | ---: | ---: | ---: | --- |
| `moge_2`          | fp16      | 19.73   | 19.72 | 50.68 | point map + metric_scale |
| `metric_anything` | fp16      | 67.00   | 66.99 | 14.93 | point map + metric_scale |

**Input 672x896**

| model         | precision | mean ms | p50   | fps   | output                 |
| --- | --- | ---: | ---: | ---: | --- |
| `unidepth_v2` | fp16      | 17.82   | 17.82 | 56.12 | point map + intrinsics |
| `unik3d`      | fp16      | 18.35   | 18.34 | 54.51 | point map + intrinsics |

**Input 1536x1536**  (only model at this size)

| model       | precision | mean ms | p50    | fps  | output |
| --- | --- | ---: | ---: | ---: | --- |
| `depth_pro` | fp16      | 242.12  | 242.17 | 4.13 | metric |

**Input 384x512**  (only model at this size)

| model      | precision | mean ms | p50  | fps    | output                   |
| --- | --- | ---: | ---: | ---: | --- |
| `zipdepth` | fp16      | 2.77    | 2.68 | 361.40 | relative (inverse depth) |

**Input 434x560**  (only model at this size)

| model  | precision | mean ms | p50   | fps   | output                    |
| --- | --- | ---: | ---: | ---: | --- |
| `tr2m` | fp16      | 20.08   | 20.08 | 49.80 | metric (text-conditioned) |

**Input 616x1064**  (only model at this size)

| model         | precision | mean ms | p50   | fps   | output                       |
| --- | --- | ---: | ---: | ---: | --- |
| `metric3d_v2` | fp32      | 62.39   | 62.39 | 16.03 | canonical (not metres - D12) |

Speeds do **not** compare across those groups: attention cost grows faster than pixel count, so a model at a smaller input is not therefore faster. Full table with p90/p99 and the per-model caveats: [reports/comparison.md](reports/comparison.md).
<!-- BENCH:END -->

---

## 라이선스

변환 스크립트는 MIT([LICENSE](LICENSE)). **업스트림 모델은 아니다.** 모델마다
`README.md` 에 코드와 가중치 각각의 라이선스 표가 있고, 둘이 다른 경우가 있다 —
Depth Anything V2 는 코드가 Apache-2.0 이지만 Base·Large 가중치는 CC BY-NC 4.0 이다.

상업적으로 쓰기 전에 그 표를 확인해라. 특히:

- **비상업용:** Uni Depth V2, UniK3D, StreamVGGT, Depth Anything V2 Base/Large 가중치
- **라이선스 파일 자체가 없음:** Depth Anything AC, TR2M — 기본적으로 어떤 사용권도 주어지지 않는다
- **자체 라이선스:** Depth Pro (Apple Sample Code), VGGT (Meta, Acceptable Use Policy 포함)

2026-07-31 에 GitHub·HuggingFace API 로 확인했다. **업스트림 LICENSE 파일이 항상
우선한다.**

---

## 문서

| | |
| --- | --- |
| [`PLAN.md`](PLAN.md) | **지금 상태와 다음 작업.** 여기부터 읽으면 된다 |
| [`docs/findings.md`](docs/findings.md) | 무엇을 재서 무엇을 알았나 — 단계별 결과와 근거 |
| [`docs/model_contracts.md`](docs/model_contracts.md) | 모델 14종의 입출력·전처리·가중치 계약, 결함 D1~D12 |
| [`docs/history.md`](docs/history.md) | 완료된 리팩토링 계획, 디렉터리 이름 변경, 재현 불가능한 역사 측정 |
| [`docs/setup.md`](docs/setup.md) | 환경 구성 상세와 알려진 함정 |
| [`tools/README.md`](tools/README.md) | 각 도구가 무엇이고 어떻게 쓰나. 끝난 질문에 답한 도구는 답까지 |
| [`docs/demo.md`](docs/demo.md) | `demo.py` 사용법 |
| [`docs/later_candidates.md`](docs/later_candidates.md) | **모델 후보 검토 기록** — 무엇을 왜 넣었고 왜 뺐나. 새 모델을 제안하기 전에 여기부터 |
