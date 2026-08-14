# tools/

루트에는 `run.py` 와 `demo.py` 만 있다 — **"모델을 돌린다" 와 "결과를 본다"**.
그 밖의 모든 실행 가능한 것이 여기 있다.

전부 **저장소 루트에서** 실행한다. 각 스크립트가 자기 위치에서 루트를 계산하므로
작업 디렉터리와 무관하게 같은 곳을 읽고 쓴다.

```bash
python tools/<이름>.py --help
```

| | |
| --- | --- |
| [1. 결과를 만드는 도구](#1-결과를-만드는-도구) | 발표되는 숫자가 여기서 나온다 |
| [2. 조사·준비 도구](#2-조사준비-도구) | 필요할 때 쓴다 |
| [3. 주장을 다시 검증하는 도구](#3-주장을-다시-검증하는-도구) | 답은 나왔지만 코드를 고치면 다시 돌려야 한다 |
| [4. `retired/`](#4-retired) | 답이 나왔고 다시 돌릴 일이 없다. 지우지 않고 보관 |

**`reports/` 아래 `.md` 는 전부 생성물이다.** 손으로 고치면 다음 생성 때 사라지고,
그 전까지는 JSON 과 어긋난 채로 발표된다 (실행 규칙 7).

---

## 1. 결과를 만드는 도구

### `compare.py` — 속도 비교표

`reports/bench/*.json` → `reports/comparison.md` + 루트 README 와 모델별 README 의
`<!-- BENCH -->` 블록.

```bash
python tools/compare.py            # 생성
python tools/compare.py --check    # 쓰지 않고, 낡았으면 exit 1
python tools/compare.py --no-readme  # comparison.md 만
```

`--check` 는 커밋 전에 돌린다. **표에 손으로 넣은 숫자를 정확히 이걸로 잡았다** —
`depth_anything_v2` 행이 세 문서에서 `4.30 / 232.53` 으로 낡아 있었다.

입력 크기별로 묶어서 낸다. 세 모델이 518 에서 돌지 않고 attention 비용은 픽셀
수보다 빠르게 늘기 때문에, 하나로 줄 세우면 "입력이 가장 작은 모델" 목록이 된다.

### `evaluate_gt.py` — 정답 깊이 데이터 채점

측정된 깊이(DIODE)로 엔진을 채점한다. `reports/gt/*.json` → `reports/gt.md`.

```bash
python tools/evaluate_gt.py --manifest data/eval/diode_indoors.json
python tools/evaluate_gt.py moge_2 --diagnose      # 한 모델, 진단 출력
python tools/evaluate_gt.py --rerender             # GPU 없이 커밋된 JSON 에서 표만 재생성
python tools/evaluate_gt.py --scale-only           # vggt·streamvggt 를 별도 표로
python tools/evaluate_gt.py --check                # 어댑터를 reports/inputs/*.npy 와 대조
```

모델마다 **어댑터**(전처리 + 깊이 추출)가 있고, `--check` 가 그 어댑터를
`reports/inputs/<model>.npy` 와 대조한다. **대조를 통과한 모델만 채점된다.**

정렬 방식이 모델의 출력 계약에서 나온다 — metric 은 `none`(적합 없음),
relative 는 `scale_shift`, 정규화 좌표는 `--scale-only`. **표에서 세로로 읽으면
안 되는 이유가 이것이고, 그래서 정렬 열이 표 안에 있다.**

### `verify_accuracy.py` — 엔진이 자기 ONNX 를 재현하는가

같은 입력을 엔진과 ONNX(fp32)에 넣어 비교한다. → `reports/accuracy.json` · `.md`

```bash
python tools/verify_accuracy.py                # 전부
python tools/verify_accuracy.py depth_pro      # 하나
python tools/verify_accuracy.py --from-json    # 재측정 없이 표만 재생성
```

입력은 `reports/inputs/<model>.npy` — **엔진을 만든 그 스크립트가 실제로 넣은
바이트**다. 다시 만들지 않으므로 이 비교가 무엇과 무엇의 비교인지 흔들리지 않는다.

### `models.py` — 재고 조사

무엇이 있고, 무엇이 빌드됐고, 무엇이 측정됐나. `spec.json` 과 `reports/` 만 읽으므로
**CUDA 없는 기계에서도 돈다.**

```bash
python tools/models.py            # 전부
python tools/models.py --stale    # 빠진 것만
python tools/models.py moge_2     # 하나
```

### `package_artifacts.py` — 배포 단위 조립

`artifacts/<model>/<profile>-<precision>/` 에 엔진·spec·manifest 를 모은다.

```bash
python tools/package_artifacts.py --verify   # 복사하지 않고 대조만
```

엔진만 배포하면 리사이즈 규칙 같은 메타데이터가 유실된다. manifest 가 그것을 같이 나른다.

---

## 2. 조사·준비 도구

### `profile_model.py` — 시간이 어디로 가는가

벤치마크가 잰 **그 엔진**에 **그 입력**으로 TensorRT 레이어 프로파일러를 붙인다.
→ `reports/profile/<model>.json`

```bash
python tools/profile_model.py moge_2 --top 20
```

**산술을 하지 않으면서 시간을 쓰는 Reformat/Shuffle 더미가 그래프 문제의
모습이다.** `moge_2` 의 이동 비중 32% 를 이걸로 찾았고, ONNX 노드를 2282→633 으로
줄여도 안 바뀌는 것도 이걸로 확인했다 — `replicate` 패딩이라 구조적이었다.

### `tune_build.py` — 빌더 설정을 바꿔 가며 빌드하고 잰다

한 모델의 그래프를 최적화 레벨·워크스페이스·정밀도를 바꿔 가며 빌드하고 각각
시간을 잰다. **발표 기록(`reports/bench`)은 절대 건드리지 않고** 엔진도
`models/<m>/engine/tune/` 에 따로 둔다. → `reports/tune/opt_<model>.json`

```bash
python tools/tune_build.py streamvggt --opt-level 4
python tools/tune_build.py streamvggt --opt-level 3 --repeats 3   # 독립 빌드 3회
python tools/tune_build.py unik3d --precision fp32
```

**`--repeats` 는 타이밍 캐시를 분리해 진짜 독립 빌드를 돌린다.** 이게 P6 을
결정했다 — 레벨 3 을 3회 빌드하니 53.30 / 53.43 / 54.03 으로 **1.4% 가 흔들렸다.**
클럭을 고정했는데도 그렇다. 단일 빌드 차이를 이득으로 발표하면 안 되는 이유가
이 숫자다 (실행 규칙 5).

**답(P6): 14개 중 `streamvggt` 하나만 레벨 4 를 쓴다.** 나머지는 기본 레벨 3.
레벨 5 는 큰 트랜스포머에서 폭발한다 — `streamvggt` +131%, `vggt` +92%.

### `size_sweep.py` — 입력 크기가 정확도를 바꾸는가

크기별로 ONNX 를 다시 내보내고 빌드해 정답 데이터로 채점한다.
**발표 기록(`reports/bench`, `reports/inputs`)은 절대 건드리지 않는다.**
→ `reports/tune/size_<model>.json`

```bash
python tools/size_sweep.py depth_anything_v2 --sizes 392x518 518x518 630x630 --dry-run
```

**답(D3): 큰 입력이 더 낫지 않다.** 세 모델 모두 원래 크기가 최선이거나 동률이고,
키우면 AbsRel 이 나빠지면서 느려진다 (dav2 23.1%→26.5%). 발표 크기를 바꾸지 않는다.

### `make_eval_manifest.py` — 평가 부분집합을 골라 적어 둔다

DIODE 다운로드(2.6 GB)는 저장소 밖에 두고 **manifest 만** 들어온다 — 어떤 파일을
어떤 순서로 골랐는지와 SHA-256.

```bash
python tools/make_eval_manifest.py --root <DIODE 경로> --split val --count 50
```

### `prepare_eval_inputs.py` — 3종횡비 크롭 재생성

`data/eval/aspects.json` 의 크롭을 원본에서 다시 만들고 sha256 을 대조한다.
manifest 는 손으로 쓰지 않는다.

```bash
python tools/prepare_eval_inputs.py
```

### `sync_desktop.sh` — 원격 GPU 기계와 동기화

노트북에는 엔진도 데이터셋도 없다. 측정은 데스크탑(RTX 3080)에서 한다.

**순서가 중요하다: 결과를 먼저 회수하고 코드를 나중에 올린다.** 반대로 하면
아직 안 가져온 측정을 덮어쓴다 (실행 규칙 3).

---

## 3. 주장을 다시 검증하는 도구

답은 이미 나왔다. 그런데 **관련 코드를 고치면 다시 돌려야 한다** — 그래서
`retired/` 가 아니라 여기 있다.

### `check_diode_convention.py`

DIODE 가 z-depth 를 저장하는가, 광선 방향 거리를 저장하는가.
**답: z-depth. 배열을 그대로 쓴다** (2026-08-14 측정).

어떤 DIODE 문서도 이걸 명시하지 않아 평면 피팅으로 직접 쟀다. 160px 지점에서
두 해석이 10.3배 갈린다 — 틀리면 모든 metric 점수가 조용히 틀어진다.

### `check_pointmap.py`

`core/pointmap.py` 가 실제 point map 위에서 MoGe 자신의 solver 와 일치하는가.
**답: 최대 3.6e-05 일치.**

```bash
python tools/check_pointmap.py --limit 5 --tol 1e-4
```

**합성 카메라 단위 테스트로는 부족하다** — 알고리즘이 맞는지는 보지만 업스트림과
같은 답을 내는지는 못 본다. 재구현 2차 시도가 좁은 최솟값을 건너뛰어 δ1 을
1.3pp 움직였고, 이 도구가 *잔차*를 비교하도록 고친 뒤에야 잡혔다.

---

## 4. `retired/`

**답이 나왔고 다시 돌릴 일이 없는 도구.** 지우지 않는 이유는 실행 규칙 9 —
실패·폐기를 결과에서 삭제하지 않는다. 주장은 다시 실행할 수 있어야 믿을 수 있고,
이 파일들은 여전히 저장소 안이며 그대로 실행된다.

| 도구 | 질문 | 답 | 증거 |
| --- | --- | --- | --- |
| `ab_input_dtype.py` | D2. 전처리를 그래프 안으로 옮기면(uint8 NHWC 입력) 이득인가 | **모델마다 다르다.** `depth_anything_v2` −20.7%, `vggt` −2.0%, `depth_pro` −1.6% — 뒤 둘은 2% 문턱 아래다. 전면 채택이 아니라 **변이로만** 채택 | `reports/uint8_ab/*.json` |
| `run_moge_ab.py` | (도구) MoGe 의 ONNX 내보내기 4종을 서로 간섭 없이 빌드 | 그래프당 3회, 순서를 바꿔 12회 | `reports/tune/moge_ab/ab_*_r*.json` |
| `summarize_moge_ab.py` | 내보내기 방식(Dynamo/TorchScript × 원본/simplified)이 엔진 속도를 바꾸는가 | **현재 것(Dynamo/simplified) 유지.** 최대 평균 차이 1.12% 로 문턱 아래. Reshape/Slice 를 줄여도 이동 비중(32.95–34.28%)이 안 줄었다 | `reports/tune/moge_ab/README.md` |
| `vis_ply.py` | (17줄) PLY 뷰어 | `demo.py` 가 PLY 를 직접 낸다 | — |
| `gen_video2imgs.py` | 영상을 프레임으로 자른다 | 이 저장소의 작업 흐름과 무관. 2025-08 이후 방치 | — |

---

## 루트에 남은 둘

`run.py` 는 모델 스크립트를 `spec.json` 이 지정한 conda 환경에서 실행하고,
`demo.py` 는 결과를 그린다. 그 둘이 이 저장소로 하는 일의 전부다.
