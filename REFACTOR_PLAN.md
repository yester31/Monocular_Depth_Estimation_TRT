# Monocular_Depth_Estimation_TRT 리팩토링 계획 (v3)

2026-07-31 / v1(`REFACTOR_DESIGN.md`) 폐기·삭제, Codex 검토(v2), Phase 0 조사(v3) 반영.

## 문서 구성

| 문서 | 역할 |
| --- | --- |
| **이 문서** | 리팩토링 계획 — 구조·단계·검증·실행 환경 |
| `docs/model_contracts.md` | **Phase 0 조사 결과** — 모델 12종의 입출력·전처리·가중치·프로필, 결함 D1~D11 |
| `CLEANUP_REVIEW.md` | 저장소 전반 정리 항목 (`get_engine` 중복, cuda-python, `results/` 등) |

## v1에서 바뀐 것

Codex 검토에서 v1의 전제 상당수가 실제 코드와 맞지 않는 것으로 드러났고, 직접 확인한 결과 사실이었다.

| v1 | v2 | 사유 |
| --- | --- | --- |
| 518 고정을 "결함"으로 취급 | **의도된 `bench` 프로필** | 모델 간 속도 비교를 위한 설계 선택. `native`와 병행 지원 |
| 입력 크기 = 단일 값 | **`native` / `bench` 2프로필** | 권장 조건 성능과 비교 조건 성능을 둘 다 본다 |
| 동적 shape 검토 | **폐기, 전부 정적 shape** | 이전 작업에서 대부분 모델이 동적 export/빌드 실패 |
| 전 모델 `uint8 [1,H,W,3]` 입력 | **프로필별 선택**, 기본은 현행 유지 | VGGT/StreamVGGT는 5차원, MoGe-2는 입력 2개. 단일 계약 불가 |
| 리사이즈 규칙을 ONNX `metadata_props`에 | **artifact manifest 별도 파일** | 엔진만 배포하면 메타데이터가 유실됨. TRT에 전달 경로 없음 |
| 모델 폴더 4파일 고정 | **capability contract** (필수+선택) | 모델별 편차가 한 파일에 안 들어감 |
| 예외를 `model.py`에 집중 | **spec(의존성 없음) / worker(torch) 분리** | `trte`가 `model.py`를 import하면 torch가 딸려와 환경 분리가 무너짐 |
| `reports/comparison.md`가 단일 출처 | **JSON/CSV가 단일 출처**, md는 생성물 | 마크다운은 기계 판독 불가 |
| "일관된 성능 비교" | **"권장 프로필 비교"** + 선택적 통제 프로필 | 해상도가 다르면 공정 비교가 아님. 명명을 바로잡음 |
| 빅뱅 재배치 | **단계적, 기존 스크립트를 oracle로 유지** | 12개 모델을 전부 실행 검증할 수 없음 |

---

## 1. 확인된 현재 상태

### 1.1 모델별 ONNX 계약 (실측)

| 모델 | 입력 이름 | rank | 출력 |
| --- | --- | --- | --- |
| depth_anything_v2 | `input` | 4D | `output` |
| depth_anything_ac | `input` | 4D | `output` |
| distill_any_depth | `input` | 4D | `output` |
| metric3d_v2 | `image` | 4D | `pred_depth` |
| depth_anything_v3 | `image` | 4D | `depth`, `sky` |
| depth_pro | `input` | 4D | `canonical_inverse_depth`, `fov_deg` |
| unik3d | `rgbs` | 4D | `pts_3d`, `confidence` |
| unidepth_v2 | `rgbs` | 4D | `pts_3d`, `confidence`, `intrinsics` |
| metric_anything | `image` | 4D | `points`, `mask`, `metric_scale` |
| **moge_2** | `image`, **`num_tokens`** | 4D + scalar | `points`, `normal`, `mask`, `metric_scale` |
| **vggt** | `images` | **5D** `[1,1,3,H,W]` | **`output_names` 주석 처리됨** |
| **streamvggt** | `images` | **5D** `[1,1,3,H,W]` | **`output_names` 주석 처리됨** |

입력 이름이 `input` / `image` / `images` / `rgbs` 4가지로 갈려 있고, 출력은 1~4개.
VGGT·StreamVGGT는 출력 이름을 지정하지 않아 ONNX가 자동 생성한 이름을 쓴다.

해상도도 518² / 616×1064 등으로 다르며, MoGe-2는 export와 실행 해상도가 불일치(Codex 지적).
**Phase 0에서 전수 조사 필요** — 위 표는 아직 미완성이다.

### 1.2 확인된 버그

| # | 내용 | 근거 |
| --- | --- | --- |
| B1 | `common_runtime.py`가 cuda-python 13에서 `ImportError` | 재현 확인. `from cuda import cuda, cudart`가 13에서 제거됨 |
| B2 | **MoGe-2는 입력이 2개인데 `inputs[0]`만 채움** | `moge_2/onnx2trt.py:148`. `num_tokens` 버퍼가 초기화되지 않은 채 실행됨 |
| B3 | fp16 버퍼를 2바이트로 할당 후 float32로 view | `common_runtime.py:45~53`. 할당 크기의 2배를 참조 |
| B4 | `allocate_buffers`가 바인딩 이름을 하드코딩 | `common_runtime.py:109~123` (`pts_3d`, `confidence`, `points`…) |
| B5 | `get_engine`이 파일 존재만으로 엔진 로드 | ONNX·빌더 옵션·TRT 버전·GPU가 바뀌어도 낡은 엔진을 씀 |
| B6 | `parser.parse_from_file()` 반환값 미확인 | 파싱 실패가 조용히 넘어감 |
| B7 | Torch 경로와 TRT 경로의 전처리가 다름 | DA V2 metric: Torch는 종횡비 유지, TRT는 518² stretch. DA AC도 동일 유형 |
| B8 | `SPARSE_WEIGHTS`가 효과 없이 켜져 있음 | 10/12에서 활성, VGGT·StreamVGGT만 주석 처리 |

**B7이 가장 무겁다.** 기존 TRT 수치와 Torch 수치는 애초에 같은 전처리가 아니었다.
따라서 리팩토링의 1차 검증 기준은 "기존 수치 재현"이 될 수 없다.

### 1.3 중복

| 항목 | 규모 |
| --- | --- |
| `get_engine()` | 19벌 / 9변종 / 1,096줄 — 기능·시그니처는 동일, 차이는 공백과 print 문구뿐 |
| `main()` | 57변종 / 5,538줄 — 설정이 전부 하드코딩 |
| 단순 복붙 함수 | 894줄 |

---

## 2. 설계 원칙 (수정본)

### 2.1 환경 경계

| 단계 | 환경 |
| --- | --- |
| PyTorch 추론 / ONNX 추출 | 모델별 전용 |
| TRT 빌드 / 벤치마크 / 데모 | 공용 `trte` |

**규칙**: 공용 런타임은 `{onnx, engine, manifest}`만 읽는다. 모델 폴더의 python을 import하지 않는다.
모델 고유 지식은 두 갈래로 나눈다.

- **spec** (`spec.json`) — 의존성 없는 선언. 공용 런타임이 읽음
- **worker** (`export.py`, `reference.py`) — torch·업스트림 import. 모델 환경에서만 실행

v1의 "예외를 `model.py`에 집중"은 이 경계를 위반하므로 폐기한다.

### 2.2 배포 단위

```
artifacts/<model>/<profile-id>/
  model.engine
  manifest.json      # 입력 레이아웃·dtype, 리사이즈/역변환 규칙, 정규화,
                     # 출력 의미, onnx·engine SHA-256, TRT/CUDA/GPU 버전
  build-report.json  # 빌드 시간, 엔진 크기, 빌더 옵션, 로그 요약
```

ONNX `metadata_props`에 의존하지 않는다. 엔진 단독 배포를 허용하지 않는다.

### 2.3 모델 폴더 계약

파일 개수를 고정하지 않는다. 대신 인터페이스를 고정한다.

| 구분 | 파일 |
| --- | --- |
| 필수 | `spec.json`, `export.py`, `reference.py`, `README.md` |
| 선택 | `wrappers.py`, `postprocess.py`, `geometry.py` 등 |

`spec.json`이 담을 것: canonical profile(해상도·rank·layout·dtype), 추가 입력, 출력 의미,
후처리 필요 여부, valid mask 규약, reference precision, conda 환경 이름, capability 플래그.

### 2.4 전처리

**기본은 현행 유지**(`float32 NCHW`, 호출 측 전처리). uint8 NHWC 내장은 **프로필 옵션**으로 두고,
아래 3개 모델 A/B 측정에서 이득이 확인된 경우에만 채택한다.

- `depth_anything_v2` — 단순 단일 출력
- `depth_pro` — 대형 입력·복수 출력
- `vggt` — 5차원 시퀀스 입력

측정 항목은 GPU compute / H2D / D2H / 전체 host latency를 **분리해서** 본다.
uint8은 전송량이 줄지만 `Cast → Transpose → Normalize`가 첫 레이어와 융합된다는 보장이 없다.

### 2.5 벤치마크

- 결과의 단일 출처는 **JSON**. `reports/comparison.md`는 그것으로부터 생성되는 산출물
- 표를 둘로 나눈다
  - **recommended profile** — 각 모델 권장 해상도. 실제 배포 비용
  - **controlled profile** — 지원 가능한 모델만 동일 입력. 선택적
- FPS 단독 순위 금지. 입력 픽셀 수·rank·프레임 수를 함께 표기
- latency 3종 분리: GPU compute / 전송 포함 / 후처리 포함
- 후처리 포함 여부를 모델마다 명시 (현재 depth_pro는 루프 안에서 FOV 후처리까지 수행)
- `builder_optimization_level` **자동 하향 금지**. 재현성을 깨뜨린다. 명시적 인자로만

### 2.6 정확도 검증

`max|diff|` 하나로는 부족하다. 출력 종류별로 지표를 나눈다.

| 출력 종류 | 지표 |
| --- | --- |
| depth / disparity | AbsRel, RMSE, δ1 (valid mask 적용) |
| point map | 점별 L2, valid mask |
| confidence / mask | IoU, 임계값 일치율 |
| scalar (fov, focal, scale) | 절대·상대 오차 |

reference precision을 `spec.json`에 명시한다 — VGGT/StreamVGGT는 export가 autocast fp16을 쓰므로
"fp32 기준선"이라는 말이 성립하지 않는다.

---

## 3. 단계별 계획

가치가 크고 위험이 낮은 것부터. **각 Phase는 독립 커밋 가능.**

### Phase 0 — 사실 조사 (코드 변경 없음)

12개 모델 전수 조사해 §1.1 계약표를 완성한다.
해상도, layout, dtype, 추가 입력, 출력 의미, 후처리 내용, valid mask 규약,
conda 환경 이름, Torch/TRT 전처리 차이(B7 범위).

산출물: `docs/model_contracts.md`

**이게 없으면 나머지 Phase의 설계 근거가 없다.**

### Phase 1 — 버그 수정 (구조 변경 없음)

B1~B6, B8. 기존 구조를 그대로 두고 고친다.

- B1: `try/except`로 cuda-python 12/13 양쪽 지원 + README에 `cuda-python` 설치 추가
- B2: MoGe-2 `num_tokens` 입력 채우기
- B3: fp16 버퍼 할당 크기 수정
- B4: 바인딩 이름 하드코딩 제거 (정적 shape이면 `trt.volume`으로 충분)
- B5: 엔진에 fingerprint(onnx SHA + 빌더 옵션 + TRT 버전 + GPU) 기록, 불일치 시 재빌드
- B6: parser 실패 시 에러 출력 후 중단
- B8: `SPARSE_WEIGHTS` 제거

위험: 낮음. **B2는 MoGe-2 출력이 실제로 바뀔 수 있으므로 전후 비교 필요.**

### Phase 2 — `get_engine()` 통합

19벌 → 루트 `common.py` 1벌. 기능 동일함은 확인됨(시그니처·플래그 일치).
Phase 1의 B5·B6·B8이 여기에 자연히 흡수된다.

검증: 각 모델 엔진을 재빌드해 **기존 엔진과 출력 바이트 비교**.

### Phase 3 — 기계 판독 결과 + 벤치마크 통일

- `reference.py`가 Torch 기준 출력을 `.npz` + 메타 JSON으로 저장
- 벤치마크 결과를 JSON으로 통일
- `compare.py`가 JSON → `reports/comparison.md` 생성
- README에 손으로 적힌 수치 전부 제거

이 시점부터 **모델 간 비교가 처음으로 성립**한다.

### Phase 4 — manifest + 구조 재배치

`spec.json` 도입, `artifacts/` 배포 단위, `models/` 하위 이동, 폴더명 소문자화,
루트 CLI(`build_engine.py` / `benchmark.py` / `compare.py` / `demo.py`).

**파괴적 변경.** 기존 경로·명령이 전부 바뀐다.
루트 README에 마이그레이션 표(구 경로 → 신 경로)를 남긴다.

### Phase 5 — uint8 전처리 (선별 적용)

§2.4의 A/B 측정 → 이득이 확인된 프로필에만 적용. 전 모델 일괄 적용 안 함.

### Phase 6 — 이후

- `later/` 17개 재시도 검토
- CPU 백엔드(ONNX Runtime / OpenVINO)
- BF16 / INT8 / CUDA Graph — Phase 3의 정확도 기준선 확보 후에만

---

## 4. 검증 전략

12개 모델을 전부 실행할 수 없다(가중치·업스트림 클론 미확보). 따라서 **단계적 golden parity**.

| 계층 | 대상 | 실행 가능성 |
| --- | --- | --- |
| L1 단위 테스트 | 리사이즈·패딩·역변환·manifest 스키마 | 항상 |
| L2 정적 검사 | ONNX checker, I/O 이름·rank·dtype, TRT parser | ONNX만 있으면 |
| L3 canary 실행 | 아래 5개 모델 | 환경 구성 필요 |
| L4 golden parity | 모델별 기존 출력 vs 신규 출력 | 모델별 |
| L5 미검증 | 나머지 | **`unverified`로 명시** |

**canary 5개** — 구조적 편차를 대표하도록 선정
`depth_anything_v2`(단일 출력) / `depth_pro`(대형 입력+scalar 출력) /
`moge_2`(다중 입출력+mask) / `vggt`(5차원+출력이름 미지정) / `metric3d_v2`(비정사각)

**규칙**
- 기존 스크립트는 golden parity 통과 전까지 **삭제하지 않는다** (oracle 역할)
- 검증 안 된 모델을 "지원 완료"로 표시하지 않는다
- 1차 기준은 "기존 수치 재현"이 아니라 **"동일 입력 텐서에서 기존 Torch와 신규 export wrapper 일치"** (B7 때문)

---

## 5. 실행 환경 — 작업은 데스크탑에서

**실제 구현·측정은 전부 3080 데스크탑에서 한다.**

| 항목 | 값 |
| --- | --- |
| 호스트 | `newpc` = `soynet` = 192.168.0.13 |
| 접속 | `ssh -i ~/.ssh/id_ed25519_codex_soy soy@192.168.0.13` (원격 셸은 cmd.exe) |
| GPU | **RTX 3080 10GB**, 드라이버 591.86 |
| CPU | i7-11700K 8C/16T, 32GB |
| conda | **`E:\APPL\anaconda3`** (24.1.2). PATH에 없으므로 전체 경로 |
| 작업 폴더 | `C:\Users\soy\mde_trt\` — `upstream\` 에 업스트림 클론 |
| 환경 위치 | **`C:\Users\soy\conda_envs\<name>`** — `conda create -p` 로 생성 |
| torch | 2.11.0+cu128, CUDA 정상 |

### 디스크 — C: 를 쓸 것

| 드라이브 | 여유 | 비고 |
| --- | ---: | --- |
| C: | 125 GB | **여기에 환경·가중치·클론** |
| D: | 32 GB | |
| E: | **4.8 GB** | conda base 가 여기 있지만 거의 가득 참 |

환경 1개(torch만) = **4.47 GB** 실측. 모델 의존성까지 5~7 GB 예상 →
12개 = 60~85 GB + 가중치 ~30 GB. C: 125 GB 로 가능하나 여유가 크지 않다.
설치하며 실측을 이어간다.

### SSH 사용 시 주의 (실측)

- 원격 셸은 cmd.exe. **`&` 로 명령을 이어붙이면 경로가 깨진다** — 한 번에 하나씩 실행
- `nvidia-smi` → `C:\Windows\System32\nvidia-smi.exe` 전체 경로
- `python`, `conda`, `git` 전부 PATH에 없음 → 전체 경로
- `wmic` 없음. `where /R` 는 매우 느림
- 여러 줄 스크립트는 `scp` 로 올린 뒤 실행하는 편이 안전

노트북(3060 6GB)이 아니라 데스크탑을 쓰는 이유: 엔진 빌드 메모리 여유, 발열 스로틀이 적어
벤치마크 신뢰도가 높음. `depth_pro`(1536²)는 6GB에서 OOM 가능성이 있다.

### 확정된 작업 분담

| | 노트북 | 데스크탑 |
| --- | --- | --- |
| 구조 리팩토링, 코드 이동 | ✅ | |
| 결함 D1~D9 수정 | ✅ | |
| `spec.json` 작성 | ✅ | |
| 리사이즈 규칙 단위 테스트 (GPU 불필요) | ✅ | |
| 문서 | ✅ | |
| 모델 환경 12개 + 업스트림 클론 + 가중치 | | ✅ **여기만** |
| ONNX export | | ✅ |
| 엔진 빌드 | | ✅ |
| 벤치마크·정확도 검증 | | ✅ **반드시** |

**근거**: 편집 도구(Edit/Write/Grep)가 로컬 파일시스템에서만 동작한다. 원격 셸이 cmd.exe라
SSH 경유 편집은 따옴표 문제로 이미 두 차례 실패했다. 반면 모델 환경은 수십 GB이고
양쪽에 구축할 이유가 없다.

### 동기화 — 네트워크 공유 우선

1. **(우선) 네트워크 공유** — 데스크탑 저장소를 `\\192.168.0.13\...` 로 마운트.
   사본이 하나뿐이라 어긋나지 않고, 로컬 편집 도구를 그대로 쓴다. 실행만 SSH.
2. **(대안) 파일 동기화** — `scp` / `robocopy` 한 줄. 공유 설정이 안 될 때.
3. git 커밋은 **실험마다가 아니라 Phase 완료 시점에만**.

### 병행 실행

**환경 구축 12종이 가장 긴 작업이고 코드 리팩토링과 독립적이다.**
데스크탑 환경·가중치 설치를 먼저 착수해 리팩토링과 겹친다.
설치는 `git clone` + `pip install` + `wget` 위주라 SSH 경유로도 따옴표 위험이 낮다.

### (선택) canary 로컬 사본

`depth_anything_v2` vits / 518² 는 6GB에 들어간다(TR2M으로 검증됨).
리팩토링 중 가장 위험한 구간에서 왕복 없는 빠른 피드백 루프를 위해 노트북에도 둘 수 있다.

## 6. 잔재로 보였던 파일 3개 — 전부 유효한 코드였다

이름이 의도를 감추고 있었을 뿐, 삭제 대상이 아니다.

| 파일 | 실제 정체 | 조치 |
| --- | --- | --- |
| `Metric_Anything/infer0.py` | **포인트클라우드·메시 출력 데모** (`save_glb`/`save_ply`/`save_plt`, 성능측정은 주석 처리) | `demo_pointcloud.py` 로 개명 |
| `VGGT/onnx_export2.py` | **3분할 엔진 export** — aggregator / depth_head / camera_head. **실제로 동작함** | `export_split.py` 로 개명 |
| `VGGT/onnx2trt2.py` | 위 3엔진 빌드·실행 | `build_split.py` 로 개명 |

`0`·`2` 접미사가 "구버전"으로 오해를 부른다. **VGGT 3분할은 README에 전혀 언급이 없다** — 문서화가 진짜 문제.

## 7. later/ 방침 (결정됨)

- **optical flow 계열 제거** — `later/WAFT` 삭제 완료 (17 → 16개)
- **나머지 16개는 보류 유지** — 루트 README에 "작업 예정" 한 줄만 추가
- 참고: 초기에 `DKT`를 optical flow로 분류했으나 오류였다.
  실제로는 *Diffusion Knows Transparency* — 투명 물체 depth/normal. 유지 대상.
  `CoTracker3`(point tracking), `GeoCalib`(camera calibration), `DINOv3`(backbone)도
  optical flow가 아니므로 유지.

## 8. spec.json 스키마 — 3축 구조

빌드 대상은 **encoder × variant × profile** 세 축의 조합이다.

| 축 | 값 | 해당 모델 |
| --- | --- | --- |
| **encoder** | vits / vitb / vitl 등 | depth_anything_v2, moge_2, unik3d, distill_any_depth, metric3d_v2 … |
| **variant** | single / split | **VGGT만** (3분할 엔진) |
| **profile** | native / bench | distill_any_depth, metric3d_v2 (native 있음) / 나머지는 bench만 |

### 확정: 스키마는 3축 전부 표현, 실제 빌드는 대표 조합만

전부 조합하면 엔진이 30개를 넘고 비교표가 흐려진다.
**비교표의 목적은 모델 간 비교**이지 모델 내 인코더 비교가 아니다.

```jsonc
{
  "name": "depth_anything_v2",
  "encoders": ["vits", "vitb", "vitl"],   // 표현은 다 하되
  "default_encoder": "vits",              // 굽는 건 이것만
  "variants": ["single"],
  "profiles": {
    "bench":  { "size": [518, 518] }
    // native 없음 — 업스트림이 가변 크기라 고정화하면 의미가 사라짐
  },
  "build_targets": [                       // 실제 빌드 목록 (명시)
    { "encoder": "vits", "variant": "single", "profile": "bench" }
  ]
}
```

`build_targets`를 명시적으로 두면, 나중에 특정 모델만 인코더를 늘리고 싶을 때
스키마를 안 건드리고 이 배열만 추가하면 된다.

### 기본 빌드 대상 (14개)

| 구분 | 수 |
| --- | ---: |
| bench 프로필 (12종 중 depth_pro 제외) | 11 |
| native 프로필 (distill_any_depth 700, metric3d_v2 616×1064, depth_pro 1536) | 3 |
| **소계** | **14** |
| VGGT split (측정용, 별도) | +2 (aggregator·camera_head) |

## 9. 미결 정리 — 전부 확정

| 항목 | 결정 |
| --- | --- |
| ~~Phase 0 계약표~~ | 완료 — `docs/model_contracts.md` |
| ~~native 기준 종횡비~~ | 불필요 — 규칙 기반 모델에는 native 미제공 |
| **환경 구성 범위** | **12개 전부.** 단 canary 5개 우선 설치 → 나머지 7개는 백그라운드.<br>최종 목표가 12종 비교표이므로 결국 전부 필요하다 |
| **마이그레이션 유예** | **즉시 전환.** 혼자 쓰는 저장소이고 외부 의존이 없다.<br>유예 장치 자체가 부채가 된다. 루트 README에 구→신 경로 매핑표 한 장만 |
| **spec.json variant 스키마** | **3축 전부 표현 + `build_targets`로 실제 대상만 명시** (위 §8) |

---

## 6. 확정 사항 (변경 없음)

- 환경 분리: ONNX 추출은 모델별, TRT는 공용 `trte`
- 리사이즈는 모델별 규칙 유지 (출력 품질 비교 목적)
- 성능 수치는 모델 README에 적지 않음 (단일 출처 원칙)
- optical flow 4개 제거 — 완료
- 모델별 라이선스 표 — 완료
