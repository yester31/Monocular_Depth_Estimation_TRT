# 모델별 계약 조사 (Phase 0)

2026-07-31 / 코드 실측. 목적은 **무엇을 공통 골격으로 묶고 무엇을 모델별로 남길지** 가르는 것.
개별 특성을 없애려는 게 아니라, 특성을 명시적으로 선언 가능한 형태로 만드는 것.

---

## 1. 입출력 계약

| 모델 | 입력 이름 | rank | 추가 입력 | 출력 |
| --- | --- | --- | --- | --- |
| depth_anything_v2 | `input` | 4D | — | `output` |
| depth_anything_ac | `input` | 4D | — | `output` |
| distill_any_depth | `input` | 4D | — | `output` |
| metric3d_v2 | `image` | 4D | — | `pred_depth` |
| depth_anything_v3 | `image` | 4D | — | `depth`, `sky` |
| depth_pro | `input` | 4D | — | `canonical_inverse_depth`, `fov_deg` |
| unik3d | `rgbs` | 4D | — | `pts_3d`, `confidence` |
| unidepth_v2 | `rgbs` | 4D | — | `pts_3d`, `confidence`, `intrinsics` |
| metric_anything | `image` | 4D | — | `points`, `mask`, `metric_scale` |
| moge_2 | `image` | 4D | **`num_tokens`** (int32 scalar) | `points`, `normal`, `mask`, `metric_scale` |
| vggt | `images` | **5D** `[1,1,3,H,W]` | — | `depth` (depth-only 래퍼) |
| streamvggt | `images` | **5D** `[1,1,3,H,W]` | — | `depth` (depth-only 래퍼) |

**입력 이름 4종** (`input`/`image`/`images`/`rgbs`), **출력 1~4개**.

VGGT·StreamVGGT는 `VGGTDepthOnlyWrapper` / `SVGGTDepthOnlyWrapper`로 depth만 남긴 변형이다.
`pose_enc`·`depth_conf`는 의도적으로 제외했고 `output_names=["depth"]`가 정상 지정돼 있다.
(바로 윗줄의 `#output_names=["pose_enc","depth","depth_conf"]`는 쓰지 않는 대안 주석 —
초기 조사에서 이 주석 줄을 실제 값으로 잘못 읽어 "미지정"으로 보고했으나 오류였다.)

### 출력 의미 분류

| 종류 | 모델 | 정확도 지표 |
| --- | --- | --- |
| relative depth | depth_anything_v2/ac, distill_any_depth | AbsRel, δ1 (스케일 정규화 후) |
| metric depth | metric3d_v2, depth_pro | AbsRel, RMSE |
| depth + 부가 | depth_anything_v3 (`sky`) | depth 지표 + mask IoU |
| point map | unik3d, unidepth_v2, metric_anything, moge_2 | 점별 L2 (valid mask 적용) |
| scalar | depth_pro (`fov_deg`), moge_2/metric_anything (`metric_scale`) | 절대·상대 오차 |
| confidence/mask | unik3d, unidepth_v2, moge_2, metric_anything | IoU |

→ **단일 `max|diff|`로는 검증 불가.** 출력 종류별 지표가 필요하다.

---

## 2. 해상도 — 업스트림 권장 vs 이 저장소

업스트림 값은 공식 저장소 코드/문서에서 확인(2026-07-31). 미확인 항목은 그렇게 표시.

**전제**: 518 통일은 **의도된 설계 선택**이다 — 모델 간 추론 성능을 비교하려면 입력을 맞춰야 한다.
따라서 아래 표의 차이는 대부분 "결함"이 아니라 "벤치마크 프로필"이다.

개선 후 목표는 **두 프로필을 모두 지원**하는 것이다.

| 프로필 | 정의 | 용도 |
| --- | --- | --- |
| `native` | 업스트림 권장 방식 그대로 | 모델의 실제 품질·권장 배포 성능 |
| `bench` | **518 고정** (현재 방식) | 모델 간 동일 조건 속도 비교 |

| 모델 | native (업스트림) | bench (현재 = 518) | native 지원 시 필요한 것 |
| --- | --- | --- | --- |
| depth_anything_v2 | 가변 — 짧은 변 518, 종횡비 유지, 14배수 | 518×518 | 동적 shape 또는 버킷 |
| depth_anything_ac | 가변 — 짧은 변 518, 종횡비 유지, 14배수 | 518×518 | 동적 shape 또는 버킷 |
| **depth_anything_v3** | **504** (`process_res=504`, `upper_bound_resize`, 504=36×14) | 518×518 | 크기·방식 변경 |
| **distill_any_depth** | **700×700** 고정, 종횡비 무시, 14배수 | 518×518 | **크기만 바꾸면 됨** ← 쉬움 |
| **depth_pro** | **1536×1536** 고정 | 1536×1536 *(bench 미적용)* | 이미 native. bench 프로필 추가 필요 |
| **metric3d_v2** | **616×1064** 고정 + mean/std 정규화 | 616×1064 *(bench 미적용)* | 이미 native(크기만). **D5** 수정 + bench 추가 |
| moge_2 | 가변 — `num_tokens`(1200~3600) + 종횡비 | 388×518 | 동적 shape 또는 num_tokens 고정 버킷 |
| metric_anything | 가변 — MoGe 계열, 내부 결정 | 518×518 | 동적 shape 또는 버킷 |
| unidepth_v2 | 가변 — 긴 변 리사이즈 + **패딩** | 518×518 **stretch** | 동적 shape + **패딩 방식 구현** |
| unik3d | 가변 — 패딩 + 픽셀 수 bound, 14배수 | 518×518 **stretch** | 동적 shape + **패딩 방식 구현** |
| vggt | 518 — crop/pad, 14배수, 패딩 **흰색** | 518×518 (1024 경유, 패딩 검정) | **D3·D7** 수정하면 native ≈ bench |
| streamvggt | 518 — crop/pad, 1024 없음, 패딩 **흰색** | 518×518 (1024 경유, 패딩 검정) | **D1·D2·D3·D7·D8** 수정하면 native ≈ bench |

### 읽는 법 — 결함과 프로필 차이의 구분

| 구분 | 항목 |
| --- | --- |
| **결함 (고쳐야 함)** | D1~D8. 업스트림과 다르게 **잘못** 구현된 것 |
| **프로필 차이 (선택)** | 518 고정, 종횡비 처리 방식 — 비교 목적의 의도된 선택 |
| **경계선** | `unidepth_v2`/`unik3d`의 stretch — bench 프로필에서는 정당하나, native 프로필에서는 패딩 방식이 필요 |

### 확정: 동적 shape 폐기, 전부 고정 크기

**결정 근거**: 이전 작업에서 대부분의 모델이 동적 shape으로 export/빌드에 실패했다.
(`depth_anything_v2/onnx_export.py`의 `dynamic = False # fail...` 주석이 그 흔적)

따라서 **두 프로필 모두 정적 shape**으로 간다.

| | 입력 크기 | 성격 |
| --- | --- | --- |
| `bench` | 518×518 고정 (전 모델 동일) | 모델 간 속도 비교 |
| `native` | **모델별 고정값** — 업스트림 규칙을 기준 종횡비에 적용해 산출 | 권장 조건 성능·품질 |

즉 native도 "가변"이 아니라 **"업스트림 규칙으로 계산해 못 박은 고정값"**이다.

파생 효과:
- `get_engine()`의 `dynamic_input_shapes` / optimization profile 경로는 **사용하지 않는다**
  (제거하거나, 미래를 위해 남기되 기본 비활성)
- 엔진은 `<model>_<profile>_<precision>.engine` 으로 프로필당 하나씩
- 입력 이미지가 그 크기가 아니면 **모델별 리사이즈 규칙**(§3의 A~F)으로 맞춘다.
  규칙은 유지하되 목표 크기만 고정되는 것

### 확정: native 프로필은 "크기가 이미 고정된 모델"에만 제공

업스트림 규칙이 입력 종횡비에 따라 크기를 바꾸는 모델은, 고정 크기로 굽는 순간
native의 의미가 사라진다. 그래서 **규칙 기반 모델에는 native를 만들지 않는다.**

| 모델 | native | bench | 근거 |
| --- | --- | --- | --- |
| **distill_any_depth** | **700×700** | 518×518 | 업스트림이 고정값. 518도 정상 동작 |
| **depth_pro** | **1536×1536** | **불가** | 아키텍처 제약 — 아래 참조 |
| **metric3d_v2** | **616×1064** | 518×518 *(속도 전용)* | metric 정확도는 깨짐 — 아래 참조 |
| depth_anything_v2 | — | 518×518 | 짧은 변 518 규칙 → 입력마다 크기 가변 |
| depth_anything_ac | — | 518×518 | 〃 |
| depth_anything_v3 | — | 518×518 | `process_res=504` upper_bound → 가변 |
| moge_2 | — | 518×518 | `num_tokens` + 종횡비 → 가변 |
| metric_anything | — | 518×518 | 〃 |
| unidepth_v2 | — | 518×518 | 긴 변 + 패딩 → 가변 |
| unik3d | — | 518×518 | 〃 |
| vggt | — | 518×518 | 업스트림도 518. **bench가 곧 native** |
| streamvggt | — | 518×518 | 〃 |

정리하면:

- **native 엔진을 따로 굽는 모델: 3종** (distill_any_depth, depth_pro, metric3d_v2)
- **vggt / streamvggt는 업스트림 자체가 518**이므로 결함(D1·D2·D3·D7·D8) 수정 후
  bench가 곧 native가 된다 — 별도 프로필 불필요
- **나머지 7종은 bench만** 제공. README에 "업스트림은 가변 크기이며 이 저장소는
  비교를 위해 518 고정을 쓴다"고 명시

### bench(518)를 적용할 수 없는 모델 — 확인됨

**`depth_pro` — 구조적 불가**

업스트림 `network/encoder.py`는 `patch_size = 384` 고정에, 입력 피라미드를 100%/50%/25%로 만든다.
`1536 = 384 × 4` 이므로 피라미드가 `1536 → 768 → 384`가 되어 최소 스케일이 정확히 패치 하나다.

518을 넣으면 `518 → 259 → 129`가 되는데 **259·129 < 384**라 슬라이딩 윈도우가 성립하지 않는다.
코드에 `assert`는 없지만 실질적으로 동작 불가.
→ **bench 프로필 없음.** 비교표에 1536으로 별도 표기하고 픽셀 수(8.79×)를 병기한다.

**`metric3d_v2` — 돌아가지만 metric 정확도가 깨짐**

canonical camera space 방식이다. 업스트림은 `img_size=(512,960)`, `focal_length=1000`인
가상 카메라로 정규화한 뒤 예측하고, 실제 intrinsics로 되돌린다.
입력 크기를 바꾸면 그 정규화 전제가 흔들려 **metric 스케일이 틀어진다.**
→ **속도 비교용으로만 bench(518) 제공.** 정확도 비교에서는 native(616×1064)만 유효.

### 작업량

| 항목 | 값 |
| --- | --- |
| 총 엔진 수 | 11(bench) + 3(native) = **14** |
| 신규 작업 | `distill_any_depth`에 native(700) 추가 |
| | `metric3d_v2`에 bench(518) 추가 — **속도 전용 표기 필수** |
| 불가 | `depth_pro` bench |

### 결론 — "전 모델 518 통일 비교표"는 성립하지 않는다

`depth_pro`가 구조적으로 빠지므로, 비교표를 둘로 나눈다.

| 표 | 내용 |
| --- | --- |
| **속도 비교** | 518 가능한 11종. `depth_pro`는 1536으로 별도 행 + 픽셀 수 병기 |
| **품질 비교** | 각 모델 native 조건. `metric3d_v2`는 여기서만 metric 유효 |

### 우선순위 제안

1. **`distill_any_depth`** — 700×700 고정이라 native 추가가 가장 쉽다. 첫 시범 대상
2. **`metric3d_v2`** — bench(518) 추가 (속도 전용 표기 필수)
3. 나머지는 결함 수정이 먼저

### 픽셀 수 비교 (FPS 해석용)

| 모델 | 입력 픽셀 | 518²=1.0 기준 |
| --- | ---: | ---: |
| 518×518 (9개 모델) | 268,324 | 1.00× |
| moge_2 (388×518) | 200,984 | 0.75× |
| metric3d_v2 (616×1064) | 655,424 | 2.44× |
| **depth_pro (1536×1536)** | **2,359,296** | **8.79×** |

**depth_pro를 다른 모델과 FPS로 직접 비교하면 오해가 생긴다.** 비교표에 픽셀 수 병기 필수.

---

## 3. 전처리 규칙 — 6가지 유형

여기가 핵심이다. 공통화 대상은 **유형**이고, 파라미터는 모델별로 남는다.

### 유형 A — keep_ratio + 14배수 (짧은 변 기준)
`depth_anything_ac`
```python
scale = 518 / min(h, w)
new_h = ceil(h*scale / 14) * 14 ;  new_w = ceil(w*scale / 14) * 14
INTER_CUBIC → ImageNet mean/std
```
⚠️ **`/255`가 없다.** 0~255 값에 ImageNet mean/std(0~1 기준)를 적용 — 버그로 보인다.

### 유형 B — keep_ratio + 14배수 (긴 변 기준, `constrain_to_multiple_of`)
`depth_anything_v2`, `distill_any_depth`
```python
scale = max(518/h, 518/w)          # v2: 더 큰 쪽 채택
new_h = constrain_to_multiple_of(scale*h, min_val=518)
INTER_CUBIC → /255 → ImageNet mean/std
```
`distill_any_depth`는 같은 함수를 쓰되 h·w 스케일을 독립 적용 (종횡비 미유지).

### 유형 C — 리사이즈 없음 (호출 측이 이미 맞춤)
`depth_anything_v3`
```python
BGR→RGB → /255 → ImageNet mean/std
```

### 유형 D — keep_ratio + 중앙 패딩 + 언패딩
`metric3d_v2`
```python
scale = min(616/h, 1064/w)
INTER_LINEAR → 중앙 패딩 (padding=[123.675,116.28,103.53])
pad_info 보관 → 출력에서 언패딩
```
정규화가 없다. 패딩 값이 ImageNet mean×255다.

### 유형 E — 정사각 패딩 + **2단 리사이즈** + 좌표 역변환
`vggt`, `streamvggt`
```python
max_dim = max(w,h) → 중앙 패딩(0)
→ 1024×1024 (INTER_CUBIC)          # target_size = 1024
→ 518×518 (F.interpolate bilinear) # 모델 입력
/255 만 (정규화 없음)
original_coords = [x1,y1,x2,y2,w,h]  # ← 1024 좌표계에서 계산됨
```
`load_and_preprocess_images_square(target_size=1024)`를 업스트림에서 가져왔는데
이 저장소는 518로 export했기 때문에 다운스케일 단계가 덧붙었다. **결함 D1~D3 참조.**

### 유형 F — stretch + 스케일 팩터 보관
`unik3d`, `unidepth_v2`
```python
cv2.resize(raw, (518,518))          # 종횡비 무시
resize_factors = (w/518, h/518)     # intrinsics 보정용
/255 → ImageNet mean/std
```
`unidepth_v2`는 `postprocess_intrinsics(intrinsics, resize_factors)` 필요.

### 기타
- `moge_2`, `metric_anything` — `/255`만, 정규화 없음. `metric_anything`은 `resize_mode` 0/1/2 선택식
- `depth_pro` — 정규화 `[0.5,0.5,0.5]`, 리사이즈는 조건부

### 정규화 요약

| 방식 | 모델 |
| --- | --- |
| ImageNet mean/std | depth_anything_v2/ac/v3, distill_any_depth, unik3d, unidepth_v2 |
| `[0.5,0.5,0.5]` | depth_pro |
| `/255`만 | moge_2, metric_anything, vggt, streamvggt |
| 없음 (패딩값이 대신) | metric3d_v2 |

---

## 4. 후처리 필요 여부

| 모델 | 후처리 | 공용 런타임이 알아야 하는가 |
| --- | --- | --- |
| depth_anything_v2/ac/v3, distill_any_depth | 없음 (리사이즈만) | 아니오 |
| metric3d_v2 | **언패딩** (pad_info 필요) | 예 |
| vggt, streamvggt | **좌표 crop** (original_coords 필요) | 예 — 모델마다 로직 다름 |
| unidepth_v2 | **intrinsics 보정** (resize_factors 필요) | 예 |
| unik3d | 없음 | 아니오 |
| depth_pro | `canonical_inverse_depth` + `fov_deg` → metric depth 변환 | 예 |
| moge_2, metric_anything | mask 적용, `inf`/`nan` 처리, 포인트클라우드 | 예 |

**전처리에서 만든 상태(pad_info, coords, resize_factors)를 후처리가 소비한다.**
→ 전·후처리를 한 쌍으로 묶어야 한다. 따로 떼면 깨진다.

---

## 5. reference precision

| 모델 | export 시 | Torch 추론 시 |
| --- | --- | --- |
| vggt, streamvggt | **autocast fp16** | autocast bf16/fp16 (GPU 따라) |
| depth_anything_v3 | autocast, fp16 언급 | half() |
| 나머지 | fp32 | half() 옵션 |

→ **"fp32 기준선"이 성립하지 않는 모델이 있다.** spec에 명시 필요.

---

## 5.5 확정된 결함

코드로 확인 완료. 추정이 아니다. **모든 항목은 업스트림 코드와 대조해 판정했다.**

### 조사 방법에 대한 교훈

**11건 중 3건이 오진이었다** (D4, D5, D10). 모두 같은 원인이다 —
**함수 하나만 보고 판단**했고, 호출부와 export 래퍼를 보지 않았다.

| 오진 | 놓친 것 |
| --- | --- |
| D4 | `preprocess_image()` 에 `/255` 가 없지만 **호출부가 이미 한다** |
| D5 | `onnx2trt.py` 에 정규화가 없지만 **ONNX 그래프 안에 있다** (`Metric3DExportModel`) |
| D10 | README 에 wget 이 있다고 봤으나 **다른 모델과 혼동** |

셋 다 grep 기반 초기 조사의 산물이다. 이후 규칙:

1. **전체 흐름을 읽는다** — `imread` 부터 텐서까지, 함수 경계를 넘어서
2. **export 래퍼를 확인한다** — 전처리가 그래프에 들어갔을 수 있다
3. **가능하면 모델을 돌려 확인한다** — D4 는 실행해 보고서야 드러났다

### 수정 현황 (2026-08-12)

| # | 상태 | 커밋 |
| --- | --- | --- |
| D1 | **수정됨** | `8eaf8d5` — D3 수정으로 좌표가 출력 격자와 일치, 근사 불필요 |
| D2 | **수정됨** | `8eaf8d5` — 가로 crop 추가 |
| D3 | **수정됨** | `8eaf8d5` — 1024 경유 제거, 518 직접 |
| D4 | **오진 → 되돌림**<br>**후속 = 프로필로 해결** | `/255` 는 호출부가 이미 하고 있었다(오진, 되돌림).<br>실제 결함인 종횡비 stretch(6.03%)는 **bench/native 두 프로필로 해결** |
| D5 | **오진 → 되돌림** | 정규화는 `Metric3DExportModel.forward()` 로 **그래프에 이미 있다**.<br>추가하면 이중 적용. 되돌림 |
| D6 | **수정됨** | `f6ea99d` — export 388×518 로 통일 |
| D7 | **수정됨** | `8eaf8d5` — 패딩 흰색 |
| D8 | **수정됨** | `8eaf8d5` — D3 와 함께 제거 |
| D9 | **수정됨** | `f6ea99d` — 절대경로 |
| D10 | **철회 — 결함 아님** | README 에 wget 안내가 애초에 없었고(`unik3d` 와 혼동),<br>`from_pretrained` 와 `hf_hub_download` 는 같은 HF 저장소다.<br>README 문서 보완만 유지 |
| **D11** | **프로필로 해결** | `unidepth_v2`/`unik3d` 에 bench/native 스위치 추가.<br>기존 동작(bench)은 그대로, native 는 종횡비 보존.<br>**엔진은 아직 안 구움** — 환경 구축 후 실측 필요 |

### 종횡비 — bench / native 두 프로필

D4 후속과 D11 을 같은 방식으로 처리했다. 지우거나 바꾸는 대신 **둘 다 제공**한다.

| 프로필 | 크기 | 성격 |
| --- | --- | --- |
| `bench` | **518×518** (전 모델 공통) | 속도 비교용. 4:3 원본을 정사각형으로 늘림 |
| `native` | **518×700** (4:3 기준) | 업스트림 방식. 종횡비 보존 |

#### 실측 — depth_anything_ac / vits / RTX 3080 / PyTorch

| 프로필 | 입력 | 시간 | 픽셀 | 업스트림 대비 |
| --- | --- | ---: | ---: | ---: |
| `bench` | 518×518 | **14.4 ms** | 268k | **6.029%** |
| `native` | 700×518 | 24.5 ms | 363k | **0.000%** |

**native 는 업스트림과 비트 단위로 일치한다** (max\|diff\| = 0). 두 프로필이 정확히
트레이드오프 관계임이 확인됐다 — bench 는 1.7배 빠르고, native 는 정확하다.

**주의: 속도 차이가 픽셀 수에 비례하지 않는다.** 픽셀은 1.35배인데 시간은 1.70배다.
ViT 의 attention 이 토큰 수에 제곱으로 늘기 때문이다
(518² → 1369 토큰, 700×518 → 1850 토큰, 1.35배 토큰이지만 attention 은 1.83배).

→ **비교표에서 FPS 를 픽셀 수로 정규화하면 안 된다.** 해상도를 병기만 하고
`pixels/s` 같은 파생 수치는 보조 이상으로 쓰지 않는다.

대상: `depth_anything_ac`, `unidepth_v2`, `unik3d`
(셋 다 규칙 기반이라 native 크기는 기준 종횡비가 필요하다 — `data/example.jpg` 의 4:3)

`onnx_export.py` 와 `onnx2trt.py` 양쪽에 `profile` 변수가 있고 **반드시 같아야 한다.**
모델 이름에 해상도가 들어가므로 두 엔진이 공존한다.
`tests/test_profiles.py` 가 두 파일의 크기 일치를 검사한다 — moge_2 처럼
export 와 실행이 어긋나 존재하지 않는 ONNX 를 찾는 일을 막는다.

**왜 unidepth_v2·unik3d 에서 더 중요한가**: 두 모델은 point map 과 intrinsics 를
출력한다. 늘어난 입력은 그림만 왜곡하는 게 아니라 **카메라 기하를 틀리게** 만든다.
`postprocess_intrinsics()` 가 `resize_factors` 로 보정하지만, 네트워크가 이미
잘못된 비율의 이미지를 보고 추론한 뒤라 완전히 상쇄되지 않는다.

#### 실측 — unik3d / vits / RTX 3080 (원본 해상도로 되돌려 비교)

| 프로필 | 시간 | depth 오차 | points 오차 | metric 스케일 |
| --- | ---: | ---: | ---: | ---: |
| `bench` 518×518 | 51.4 ms | **215.2%** | 210.8% | 3.15× |
| `native` 518×700 | 30.0 ms | **75.2%** | 81.9% | 1.75× |

native 가 개선하지만 **여전히 크게 어긋난다.** depth_anything_ac 처럼 0% 로 떨어지지 않는다.

#### 원인 — 이 두 모델은 스스로 리사이즈한다

```
shape_constraints: pixels_min 200,000 / pixels_max 600,000
                   ratio_bounds [0.5, 2.5] / shape_mult 14
```

| 넣은 크기 | 네트워크가 실제로 본 크기 |
| --- | --- |
| 원본 3024×2268 (6.9M px) | **896×672** (602k px — 상한에 맞춰 축소) |
| 518×518 (268k px) | 518×518 (이미 범위 안 → 통과) |
| 518×700 (363k px) | 518×700 (이미 범위 안 → 통과) |

`unidepth_v2` 도 동일한 제약을 갖는다 (실측 확인).

**즉 사전 리사이즈가 모델의 자체 규칙을 가로챈다.** 우리가 518×518 로 줄여서 넣으면
모델은 "원래 이 크기의 이미지"로 알고 그에 맞는 초점거리를 추론한다:

| | fx | fy |
| --- | ---: | ---: |
| 원본 투입 (unidepth_v2) | 2859.4 | 2955.8 |
| 518×518 투입 | 551.1 | 561.4 |
| 518×700 투입 | 627.6 | 647.8 |

metric depth 는 초점거리에 묶여 있으므로 **깊이 값 자체가 달라진다.**

**결론**: 이 두 모델에서 `native` 는 "업스트림과 같은 결과"를 뜻하지 않는다.
진짜 native 는 **원본을 그대로 넣어 모델이 스스로 896×672 를 고르게 하는 것**인데,
그러면 입력 크기가 이미지마다 달라져 정적 엔진을 만들 수 없다.

#### 896×672 를 넣으면? — 개선되지만 0 은 아니다

| 넣은 크기 | 시간 | depth 오차 | metric 스케일 |
| --- | ---: | ---: | ---: |
| 원본 3024×2268 (기준) | 45.2 ms | — | 1.00× |
| **896×672** (모델이 스스로 고르는 크기) | 39.5 ms | **17.8%** | 1.18× |
| 518×700 | 29.9 ms | 75.2% | 1.75× |
| 518×518 (bench) | 27.3 ms | 215.2% | 3.15× |

896×672 가 오차를 215% → **17.8%** 로 크게 줄이지만 **0 이 되지 않는다.**
모델이 원본에서 896×672 를 고를 때와, 우리가 미리 896×672 로 줄여 줄 때가 다르기 때문이다 —
전자는 6.9M 픽셀에서 한 번에 축소하고, 후자는 이미 축소된 이미지를 받는다.
축소 과정의 정보 손실이 다르고, 모델은 그 차이를 "다른 카메라"로 해석한다.

#### 오차의 정체 — 대부분 스케일이지 구조가 아니다

위 백분율은 `mean|a-b| / mean|a|` 이므로 전체가 상수배로 밀리기만 해도 커진다.
스케일을 분리해 보면:

| 입력 | 원래 오차 | 스케일 | **스케일 보정 후** | 구조 상관 |
| --- | ---: | ---: | ---: | ---: |
| 896×672 | 17.8% | 1.18× | **0.7%** | **0.9961** |
| 518×700 | 75.2% | 1.75× | **3.3%** | 0.8968 |
| 518×518 | 215.2% | 3.15× | **5.6%** | 0.7210 |

(스케일 = 최소제곱 최적 배수, 상관 = 두 깊이 맵의 Pearson 상관계수)

**215% 가 보정 후 5.6% 로 떨어진다.** 깊이 맵의 구조는 대체로 살아 있고
전체가 상수배로 이동한 것이다.

#### 결론 — 용도에 따라 다르다

| 용도 | 판정 |
| --- | --- |
| 상대적 깊이 (어디가 가깝고 먼지) | 896×672 면 상관 0.9961 — **거의 완전** |
| metric 절대값 (몇 미터인지) | 스케일이 1.18~3.15배 틀림 — **보정 필요** |

**스케일 보정은 가능하다.** 두 모델은 intrinsics 를 함께 출력하므로 원본 대비
초점거리 비율로 깊이를 나누면 된다. `postprocess_intrinsics()` 가 이미 그 방향의
코드지만 현재는 intrinsics 에만 적용하고 depth 에는 적용하지 않는다.

**bench(518×518) 는 여전히 권장하지 않는다** — 상관 0.7210 으로 구조까지 무너진다.
종횡비를 늘린 결과다. `native` 를 **896×672** 로 잡는 것이 확실히 낫다
(상관 0.9961, 스케일 보정 후 0.7%). 다만 602k 픽셀로 518² 의 2.24배라
속도는 다른 모델과 나란히 비교할 수 없다.

**비교표 표기 의무**: "518×518 에서 잰 unik3d 의 metric depth 는 업스트림의 3.15배"
라는 사실을 모르면 수치를 오해한다. 반드시 명시할 것.

**단위 테스트**: `tests/test_vggt_geometry.py` (D1·D2·D3·D7·D8),
`tests/test_preprocess.py` (전처리 유형), `tests/test_golden.py` (검증 체계).

**아직 end-to-end 재실행은 하지 않았다.** canary 환경 구축 후 골든 기준선과 대조 예정.

### 후속 과제

`torch.load()` 가 저장소 전반에서 `weights_only` 없이 호출된다. 신뢰할 수 없는
체크포인트를 로드하면 임의 코드가 실행된다. 다만 일부 체크포인트는 텐서 외
객체를 담고 있을 수 있어 일괄 변경 시 로딩이 깨질 수 있으므로, 모델별로
확인하며 바꿔야 한다.

### 수정 방안 요약

| # | 모델 | 증상 | 업스트림은 | 수정 |
| --- | --- | --- | --- | --- |
| D1 | streamvggt | 좌표를 `/2`로 근사 (실제 1024/518=1.977) | 1024 단계 자체가 없음 | D3 수정 시 자동 해소 |
| D2 | streamvggt | 가로 crop 누락 (`, :`) | x·y 모두 crop | `[y1:y2, x1:x2]` 로 수정 |
| D3 | vggt, streamvggt | 1024 경유 후 518로 축소 | **518 직접** | 정사각 패딩 → 518 직접 리사이즈, `scale = 518/max_dim` |
| D4 | depth_anything_ac | `/255` 누락 | `.astype(np.float32) / 255.0` 후 정규화 | `/255` 한 줄 추가 |
| D5 | metric3d_v2 | mean/std 정규화 없음 | `mean=[123.675,116.28,103.53]`, `std=[58.395,57.12,57.375]` | 정규화 추가 |
| D6 | moge_2 | export 291 ≠ TRT 388 | — | 하나로 통일 (388 권장, 현재 실행값) |
| D7 | vggt, streamvggt | 패딩 검정 `[0,0,0]` | **흰색 `value=1.0`** | `value=[255,255,255]` (uint8 기준) |
| D8 | streamvggt | 1024 단계에 근거 없음 (VGGT 코드 복사) | `load_fn.py`에 square 변형 없음 | D3와 함께 제거 |
| D9 | depth_anything_ac | 상대경로 `checkpoints/` — CWD 의존 | — | `{CUR_DIR}/...` 절대경로 |
| D10 | unidepth_v2 | README wget / export `hf_hub_download` / infer `from_pretrained` 3중 불일치 | — | 하나로 통일 + README 수정 |
| D11 | unidepth_v2, unik3d | 518²로 **stretch** (종횡비 무시) | **패딩으로 종횡비 유지** | 비율 유지 리사이즈 + 패딩. 엔진 크기는 518² 유지 |

**D3이 핵심이다.** 이것만 고치면 D1·D8이 함께 사라지고, vggt/streamvggt의 native ≈ bench가 된다.

### 수정 후 검증 방법

| # | 검증 |
| --- | --- |
| D1~D3, D7 | 비정사각 이미지(예: 640×480)로 Torch vs TRT 출력 비교. 패딩 영역·crop 경계 확인 |
| D4, D5 | 정규화 전후 출력 통계(min/max/mean) 비교. 기존 결과와 크게 달라져야 정상 |
| D6 | export/실행 크기 일치 후 엔진 빌드 성공 여부 |
| D9 | 저장소 루트에서 서브프로세스로 실행해 통과 확인 |
| D10 | export/infer가 같은 가중치를 읽는지 SHA 비교 |

**주의**: D4·D5는 고치면 **출력값이 크게 바뀐다.** 기존 결과와 다르다고 잘못된 게 아니라,
지금까지가 틀렸던 것이다. 기존 벤치마크 수치는 이 시점에서 폐기해야 한다.

### D1 — StreamVGGT 좌표 역변환이 1024/518 비율을 2로 근사

`streamvggt/onnx2trt.py:178`
```python
depth = depth[int(original_coord[1]/2) : int(original_coord[3]/2), :]
```
`original_coords`는 `scale = 1024 / max_dim`으로 계산되는데(:106), 출력은 518이다.
올바른 비율은 **1024/518 = 1.977**인데 2로 나눈다 → **약 1.2% 어긋남**, 518px 기준 최대 6px 오차.

### D2 — StreamVGGT 가로 crop 누락

같은 줄의 `, :` — 세로만 자르고 **가로는 자르지 않는다.**
원본이 가로로 긴 이미지(대부분)면 좌우 패딩이 출력에 그대로 남는다.

비교: `vggt/onnx2trt.py:184-185`는 출력을 1024로 되돌린 뒤 x·y 모두 crop한다 — 이쪽이 맞다.
```python
depth = cv2.resize(depth, (1024, 1024), cv2.INTER_LINEAR)
depth = depth[int(y1):int(y2), int(x1):int(x2), ...]
```

### D3 — 1024 중간 단계는 순손실

1024로 올렸다가 518로 내리는 것은 직접 518로 리사이즈하는 것보다 나을 수 없다.
`F.interpolate(bilinear)`는 antialias가 꺼져 있어 다운샘플에서 에일리어싱이 생긴다.
프레임당 1024×1024 INTER_CUBIC 리사이즈 비용이 추가되며, 벤치마크 루프 안에 있으면
다른 모델과의 비교를 왜곡한다.

**수정 방향**: 정사각 패딩 → **518 직접 리사이즈**, 좌표도 `scale = 518 / max_dim`으로 계산.
D1이 원천적으로 사라지고, VGGT의 "518→1024 되돌리기"도 불필요해진다. D2는 별도 수정.

### D4 — ~~depth_anything_ac `/255` 누락~~ → **오진. 실제는 종횡비 stretch**

**정정 (2026-08-12).** `/255` 누락이 아니었다. `preprocess_image()` 안에는 없지만
호출부(`onnx2trt.py:71`)가 이미 `cv2.cvtColor(...).astype(np.float32) / 255.0` 을 한다.
함수만 보고 판단한 초기 조사의 오류다.

**실제 결함은 그 한 줄 위에 있다.**

```python
70:  raw_image = cv2.resize(raw_image, (input_w, input_h))   # 518x518 강제
71:  image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
73:  input_image = preprocess_image(image, input_h)          # 종횡비 유지 로직
```

70행이 종횡비를 무시하고 정사각형으로 늘린 뒤, 73행의 종횡비 유지 리사이즈가
**이미 정사각형인 이미지에 적용되어 무의미해진다.**

| | 네트워크 입력 |
| --- | --- |
| 업스트림 (`tools/infer.py`) | **518×700** (짧은 변 518, 종횡비 유지) |
| 이 저장소 | **518×518** (stretch) |

3024×2268 샘플 기준 **출력 차이 6.03%** (RTX 3080 실측).

`preprocess_image()` 자체는 업스트림과 줄 단위로 일치한다 — 70행만 제거하면 된다.
다만 이는 엔진 입력 크기를 518×518 → 518×700 으로 바꾸므로 **재export·재빌드가 필요**하고,
`bench` 프로필(전 모델 518 통일)과 충돌한다. → §2 프로필 논의와 함께 결정해야 함.

**측정 근거**
```
upstream (keep ratio -> 518x700)
repo     (stretch to 518x518)      rel 6.03%
```

### D5 — ~~metric3d_v2 정규화 누락~~ → **오진. 그래프 안에 있었다**

**정정 (2026-08-12).** `onnx2trt.py` 에 정규화가 없는 것은 맞지만, **있으면 안 된다.**

`onnx_export.py:38` 이 모델을 `Metric3DExportModel` 로 감싸고, 그 `forward()` 의
첫 줄이 `image = self.normalize_image(image)` 다 (업스트림 `onnx/metric3d_onnx_export.py`).
`mean=[123.675,116.28,103.53]`, `std=[58.395,57.12,57.375]` 가 **ONNX 그래프에 박혀 있고**,
따라서 엔진은 정규화되지 않은 0~255 입력을 기대한다.

`onnx2trt.py` 에서 정규화하면 **이중 적용**이 된다.

초기 조사에서 `onnx_export.py` 를 `mean|std` 로만 grep 해 아무것도 못 찾았고,
래퍼 클래스가 별도 파일에서 import 되는 것을 놓쳤다.

### D6 — moge_2 export/실행 해상도 불일치

`onnx_export.py:45` → `291, 518` / `onnx2trt.py:101` → `388, 518`

### D7 — VGGT·StreamVGGT 패딩 색이 업스트림과 반대

| | 패딩 값 |
| --- | --- |
| 업스트림 (VGGT `load_fn.py`, StreamVGGT `load_fn.py`) | **흰색 `value=1.0`** |
| 이 저장소 (`vggt/onnx2trt.py:99`, `streamvggt/onnx2trt.py:118`) | **검은색 `[0,0,0]`** |

업스트림 주석에도 `# Pad with white (value=1.0)`로 명시돼 있다.
정사각이 아닌 입력에서 패딩 영역이 모델에 다르게 보이므로 출력이 달라진다.

### D11 — unidepth_v2 / unik3d 가 종횡비를 무시하고 stretch

**업스트림** (`unik3d/models/unik3d.py` `infer()`, unidepth_v2도 동일 방식)
```python
paddings, (padded_H, padded_W) = get_paddings((H, W), ratio_bounds)  # 종횡비 맞춰 패딩
resize_factor, (new_H, new_W) = get_resize_factor(...)               # 14 배수
rgb = F.pad(rgb, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
```
→ **패딩으로 종횡비를 지킨다. 늘리지 않는다.**

**이 저장소** (`unik3d/onnx2trt.py:124`, `uni_depth_v2/onnx2trt.py:112`)
```python
raw_image = cv2.resize(raw_image, (input_w, input_h))   # 518x518 강제
resize_factors = (w/input_w, h/input_h)
```
→ 640×480 이면 가로가 세로보다 더 눌린다. 형태가 왜곡된다.

**왜 이 두 모델에서 특히 문제인가**

출력이 `pts_3d`(3D 점 좌표)와 `intrinsics`(카메라 내부 파라미터)다.
카메라 기하는 종횡비에 직접 묶여 있어, 늘린 이미지를 넣으면 모델이 **다른 렌즈로 찍은 사진**으로
해석한다. depth 한 장만 내는 모델과 달리 되돌리기가 간단하지 않다.

`postprocess_intrinsics(intrinsics, resize_factors)` 로 보정을 시도하고 있으나,
모델이 이미 왜곡된 입력을 보고 추론한 뒤라 완전한 상쇄는 기대할 수 없다.

**수정**

엔진 입력은 518×518 그대로 두고(재빌드 불필요) 채우는 방식만 바꾼다.
```
현재:  640×480 → 518×518 강제 (가로가 더 눌림)
수정:  640×480 → 518×389 비율 유지 → 위아래 패딩 → 518×518
```
후처리도 함께: 패딩 영역 제거, `resize_factors` 를 패딩 기준으로 재계산
(현재 값은 stretch 기준이라 어차피 맞지 않는다).

**수정 후 실측 목적**: 차이가 크면 기존 벤치마크에서 이 두 모델의 품질 수치가
부당하게 낮았다는 뜻이므로, 그 사실을 비교표에 남긴다.

### D10 — ~~unidepth_v2 가중치 경로 불일치~~ → **결함 아님**

**철회 (2026-08-12).** 두 가지를 잘못 봤다.

1. "README 의 wget 안내가 안 쓰인다" — **README 에 wget 안내가 애초에 없다.**
   초기 조사에서 `unik3d`(wget 3개 있음)와 혼동했다.
2. `infer.py` 의 `from_pretrained("lpiccinelli/unidepth-v2-{enc}14")` 와
   `onnx_export.py` 의 `hf_hub_download(repo_id="lpiccinelli/unidepth-v2-{enc}14", ...)` 는
   **같은 HuggingFace 저장소를 가리킨다.** API 만 다를 뿐 같은 가중치다.

남은 것은 문서 부재뿐이다 — README 에 가중치를 어떻게 얻는지 설명이 없었다.
설명을 추가했으나 이는 결함 수정이 아니라 문서 보완이다.

### D9 — depth_anything_ac 가 상대경로로 체크포인트를 읽는다

`depth_anything_ac/infer.py:88`
```python
checkpoint = torch.load(f'checkpoints/depth_anything_AC_{encoder}.pth', ...)
```
다른 모델은 모두 `{CUR_DIR}/...` 절대경로를 쓴다. 이 모델만 **CWD에 의존**한다.
루트 CLI에서 서브프로세스로 호출하면(`benchmark.py --backend torch`) 깨진다.

### D8 — StreamVGGT의 1024 단계는 업스트림에 없다

| | |
| --- | --- |
| VGGT 업스트림 | `load_and_preprocess_images_square(target_size=1024)` **존재** — 좌표 반환용 |
| StreamVGGT 업스트림 | `load_fn.py`에 **square 변형 없음**, `target_size = 518`만 |

`streamvggt/onnx2trt.py`의 1024 경유는 **VGGT 코드를 그대로 복사한 결과**로 보인다.
StreamVGGT에는 근거가 없다. D1(비율 2 근사)·D2(가로 crop 누락)가 여기서 파생됐다.

---

## 6. Torch 경로 vs TRT 경로 불일치 (확인됨)

| 모델 | Torch | TRT | 영향 |
| --- | --- | --- | --- |
| depth_anything_v2 (metric) | 원본 종횡비 유지 | 518² 강제 stretch 후 처리 | 출력이 다름 |
| depth_anything_ac | 짧은 변 기준 리사이즈 | 정사각 변환 | 출력이 다름 |

→ 리팩토링 1차 검증 기준을 "기존 수치 재현"으로 잡으면 **불일치를 고착**시킨다.

---

## 7. 결론 — 무엇을 통일하고 무엇을 남기는가

### 통일 가능 (공통 골격)

| 항목 | 근거 |
| --- | --- |
| `get_engine()` | 19벌 전부 기능 동일 |
| 엔진 캐시 fingerprint, parser 에러 처리 | 전 모델 공통 문제 |
| 벤치마크 루프 (워밍업·반복·타이밍) | 측정 방식은 모델과 무관 |
| 결과 JSON 스키마 | 비교하려면 필수 |
| 리사이즈 **유형** A~F 구현 | 6종으로 수렴 |
| 정규화 연산 (mean/std, /255, none) | 파라미터만 다름 |
| 시각화 (컬러맵·저장) | 출력 종류별 3~4종 |

### 모델별로 남김 (spec에 선언)

| 항목 | 이유 |
| --- | --- |
| 입출력 이름·rank·개수 | 1~4개, 4D/5D, 추가 입력 |
| 해상도 | 518² ~ 1536², 비정사각 포함 |
| 리사이즈 유형 + 파라미터 | 6종 × 파라미터 |
| 정규화 방식 | 4종 |
| 후처리 종류 | 언패딩/crop/intrinsics/metric 변환/mask |
| 정확도 지표 | 출력 종류에 따라 다름 |
| reference precision | autocast 여부 |
| conda 환경 이름 | 모델마다 다름 |

### spec.json 필드 (초안)

```jsonc
{
  "name": "metric3d_v2",
  "env": "metric3d",
  "input":  { "name": "image", "rank": 4, "layout": "NCHW",
              "dtype": "float32", "size": [616, 1064] },
  "extra_inputs": [],
  "preprocess": { "resize": "keep_ratio_pad", "interp": "linear",
                  "pad_value": [123.675, 116.28, 103.53],
                  "normalize": "none" },
  "postprocess": ["unpad"],
  "outputs": [ { "name": "pred_depth", "kind": "metric_depth" } ],
  "reference_precision": "fp32",
  "capabilities": ["depth"]
}
```

---

## 8. 셋업 계약 — 업스트림·환경·가중치

전처리만큼이나 편차가 크다. 여기도 **유형은 통일, 파라미터는 모델별**로 간다.

### 8.1 conda 환경 이름

| 모델 | 환경 이름 |
| --- | --- |
| depth_anything_v2 | `dav2` |
| depth_anything_v3 | `dav3` |
| depth_anything_ac | `depth_anything_ac` |
| depth_pro | `depth-pro` |
| distill_any_depth | `distill-any-depth` |
| metric3d_v2 | `metric3d` |
| metric_anything | `metric_anything` |
| moge_2 | `MoGe` |
| streamvggt | `streamvggt` |
| unidepth_v2 | `unidepthv2` |
| unik3d | `unik3d` |
| vggt | `vggt` |

축약형(`dav2`)·하이픈(`depth-pro`)·언더스코어(`metric_anything`)·대문자(`MoGe`)가 뒤섞여 있다.
**`spec.json`에 선언만 하면 되므로 이름을 바꿀 필요는 없다.** 다만 선언은 반드시 필요하다
— `benchmark.py --backend torch`가 어느 환경을 부를지 알아야 한다.

### 8.2 업스트림 클론 후 처리

| 유형 | 모델 | 내용 |
| --- | --- | --- |
| 그대로 사용 | 대부분 | `git clone` 후 끝 |
| **rename 필요** | depth_anything_v3 | `Depth-Anything-3` → `Depth_Anything_V3` |
| **rename 필요** | metric_anything | `metric-anything` → `metric_anything` |

하이픈이 python 패키지명으로 못 쓰이기 때문. spec에 `post_clone_rename` 필드가 필요하다.

### 8.3 `sys.path` 삽입 경로

| 모델 | 경로 |
| --- | --- |
| depth_anything_v2 | `Depth-Anything-V2` |
| depth_anything_ac | `DepthAnythingAC` |
| metric3d_v2 | `Metric3D` |
| streamvggt | `StreamVGGT/src` ← 하위 디렉터리 |
| metric_anything | `metric_anything/models/student_pointmap` ← 3단계 하위 |

깊이가 제각각. spec에 `import_path`로 선언.

### 8.4 가중치 획득 방식 — 5가지

| 방식 | 모델 | 특징 |
| --- | --- | --- |
| **A. wget → 로컬 경로** | depth_anything_v2(9개 URL), distill_any_depth(4), moge_2(4), unik3d(3), metric_anything(2), depth_anything_v3(2), streamvggt(1) | URL 명시. 오프라인 가능 |
| **B. 업스트림 스크립트** | **depth_pro** — `source get_pretrained_models.sh` → `https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt` → `checkpoints/` | README에 URL이 없던 게 아니라 스크립트가 감싸고 있었다. spec에는 URL을 직접 적으면 됨 |
| **C. torch.hub 자동** | **metric3d_v2** — `metric3d_vit_small(pretrain=True)` 한 줄로 자동 다운로드<br>**vggt** — `torch.hub.load_state_dict_from_url(_URL)` | 별도 다운로드 단계가 아예 없음. **정상 동작** |
| **D. HF 직접** | unidepth_v2 — export는 `hf_hub_download`, infer는 `from_pretrained` | **경로 불일치 → D10** |
| **E. HF CLI** | depth_anything_ac — `huggingface-cli download` | |

**정정**: 초기 조사에서 `depth_pro`·`metric3d_v2`를 "수동 다운로드(README에 URL 없음)"로 분류했으나
오류였다. 전자는 업스트림 스크립트, 후자는 torch.hub 자동이며 둘 다 정상이다.

**unik3d 도 정상이다** — README의 wget으로 받은 로컬 safetensors를 `if 1:` 분기에서 실제로 쓴다.
`from_pretrained`는 `else:` 쪽 죽은 분기.
실제로 어긋나는 것은 **unidepth_v2 하나뿐**(D10).

### 8.5 체크포인트 저장 위치

| 패턴 | 모델 |
| --- | --- |
| `<model>/checkpoints/` | depth_anything_ac, metric_anything |
| `<upstream>/checkpoints/` | depth_anything_v2, depth_pro |
| `<upstream>/checkpoint/<variant>/` | distill_any_depth, moge_2 |
| `<upstream>/depth-anything/DA3METRIC-LARGE/` | depth_anything_v3 |
| `<model>/ckpt/` | streamvggt |
| `checkpoints/<variant>/` | unik3d |
| HF 캐시 / torch hub 캐시 | unidepth_v2, vggt |

**모델 폴더 기준 / 업스트림 폴더 기준 / 전역 캐시**가 섞여 있다.
`depth_anything_ac/infer.py`는 상대경로 `checkpoints/...`를 써서 CWD에 의존한다 — 루트에서 실행하면 깨진다.

### 8.6 requirements

| 모델 | 파일 |
| --- | --- |
| 대부분 | `requirements.txt` |
| metric3d_v2 | `requirements_v2.txt` (+`_v1`) |
| vggt | `requirements.txt` + `requirements_demo.txt` |
| metric_anything | `models/student_pointmap/requirements.txt` |
| depth_pro, depth_anything_v3 | 명시 없음 (`pip install -e .` 등) |

### 8.7 → spec.json 셋업 필드 (추가안)

```jsonc
{
  "setup": {
    "upstream": "https://github.com/DepthAnything/Depth-Anything-V2",
    "clone_dir": "Depth-Anything-V2",
    "post_clone_rename": null,
    "import_path": "Depth-Anything-V2",
    "env": "dav2",
    "requirements": ["Depth-Anything-V2/requirements.txt"],
    "weights": [
      { "method": "url",
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "dest": "Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth",
        "variant": "vits", "license": "apache-2.0" },
      { "method": "url", "...": "vitb / vitl (CC BY-NC 4.0)" }
    ]
  }
}
```

`method`는 `url` / `hf_repo` / `hf_file` / `torch_hub` / `manual` 5종.
공통 다운로더가 `method`별 처리기를 갖고, 모델은 선언만 한다.

이러면 `python setup_model.py depth_anything_v2` 한 줄로 클론·환경·가중치가 끝나고,
**모델별 차이는 전부 spec 안에 남는다.**

---

## 9. 조사 결과 — 미확인 없음

확정 결함은 §5.5(**D1~D11**). 코드 조사로 답할 수 있는 항목은 전부 조사를 마쳤다.

| 항목 | 결론 |
| --- | --- |
| 동적 shape 실현 가능성 | **폐기** — 전부 정적 shape (사용자 결정) |
| 업스트림 권장 사이즈 12종 | 전부 확인 (각 저장소 데모/추론 코드) |
| VGGT·StreamVGGT 출력 이름 | `output_names=["depth"]` 정상. depth-only 래퍼. **초기 "미지정" 보고는 내 추출 스크립트 오류** |
| `depth_pro` 조건부 리사이즈 | 종횡비 무시 stretch. bench(518)는 아키텍처상 불가 |
| `depth_pro` 가중치 | `get_pretrained_models.sh` → `https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt`. 정상 |
| `metric3d_v2` 가중치 | `metric3d_vit_*(pretrain=True)` → torch.hub 자동. 정상 |
| `unik3d` 가중치 | 로컬 safetensors 사용(`if 1:` 분기). 정상 |
| `unidepth_v2` 가중치 | README·export·infer 3중 불일치 → **D10** |
| `depth_anything_ac` CWD 의존 | → **D9** |
| stretch 왜곡 | 업스트림은 패딩 → **D11** |

### 오차를 어떻게 재는가

`core/golden.py` 의 `compare()`:

```python
finite   = np.isfinite(a) & np.isfinite(b)   # NaN/Inf 는 쌍으로 제외
d        = np.abs(a[finite] - b[finite])
max_abs  = d.max()
mean_abs = d.mean()
rel_mean = d.mean() / np.abs(a[finite]).mean()   # 표의 "오차"
```

**`평균 절대차 ÷ 기준값의 평균 크기`** 다. 주의할 점:

- **100% 를 넘을 수 있다.** 픽셀별 상대오차가 아니라 전체 크기 대비 차이라서,
  값이 3배가 되면 200% 가 나온다. 그래서 표에 스케일 배수를 함께 적는다.
- **`mean(|a-b|/|a|)` 가 아니다.** 그 형태는 깊이가 0 에 가까운 픽셀에서 발산한다.
  `unik3d` 는 `depth.min() == 0` 이라 쓸 수 없다.
- **스케일 오차와 구조 오차를 구분하지 못한다.** 전체가 상수배로 밀려도 큰 값이 나온다.
  구분이 필요하면 최소제곱 스케일을 제거한 뒤 다시 재고, Pearson 상관으로 구조를 본다
  (unik3d 절에 그 예가 있다).
- NaN/Inf 를 쌍으로 제외한다. `moge_2`·`metric_anything` 은 유효하지 않은 픽셀을
  `inf` 로 표시하므로, 그러지 않으면 한 픽셀이 전체 통계를 삼킨다.

### 실측이 필요한 항목 (데스크탑, 결함 수정 후)

조사로는 답이 안 나오고 **돌려봐야 아는 것들**이다. 미결 사항이 아니라 측정 대상.

| 항목 | 목적 |
| --- | --- |
| D11 수정 전후 차이 | stretch → 패딩 전환이 `pts_3d`·`intrinsics`를 얼마나 바꾸는지. 차이가 크면 기존 비교표에서 이 두 모델이 부당하게 낮게 평가됐다는 뜻 |
| D4·D5 수정 전후 차이 | 정규화 추가로 출력이 얼마나 바뀌는지. **기존 벤치마크 수치 폐기 근거** |
| fp16 vs fp32 정확도 | 전 모델. **한 번도 측정된 적 없음** |
| native vs bench | `distill_any_depth`(700 vs 518), `metric3d_v2`(616×1064 vs 518) 성능·품질 |
| **VGGT single vs split** | 단일 엔진(depth만) vs 3분할 엔진(depth+conf+pose). **어느 쪽이 빠른지 미측정** — 아래 참조 |

### VGGT 두 가지 운용 방식 — 측정 필요

`VGGT/onnx_export2.py` + `onnx2trt2.py`는 실험 잔재가 아니라 **실제로 동작하는 대안 경로**다.

| 변형 | 엔진 | 출력 |
| --- | --- | --- |
| `single` (`onnx_export.py`) | 1개 `vggt_only_depth_518x518` | `depth` |
| `split` (`onnx_export2.py`) | 3개 `vggt_aggregator` / `vggt_depth_head` / `vggt_camera_head` | `depth`, `depth_conf`, `pose_enc` |

split은 기능이 더 많고(confidence·camera pose), aggregator를 한 번 돌린 뒤 head만 갈아끼울 수 있어
여러 출력을 쓸 때 구조적으로 유리하다. 반면 엔진 3개를 순차 실행하므로 오버헤드가 있다.

**README에 이 방식이 전혀 언급돼 있지 않고 성능도 측정된 적이 없다.**

측정 결과에 따라:
- split이 더 빠르거나 비슷하면 → **split을 기본**으로
- 느리면 → "출력이 더 필요할 때 쓰는 선택지"로 문서화

**계획에 미치는 영향**: "모델 하나 = 엔진 하나" 전제가 깨진다.
`spec.json`이 **variant** 개념을 가져야 한다. (StreamVGGT에는 split 구현이 없어 VGGT만 해당)

### 조사 출처 (전부 업스트림 데모/추론 코드)

| 모델 | 확인한 파일 |
| --- | --- |
| depth_anything_v2 | `depth_anything_v2/dpt.py` — `infer_image(input_size=518)` |
| depth_anything_ac | `tools/infer.py` |
| depth_anything_v3 | `src/depth_anything_3/api.py` — `inference(process_res=504)` |
| distill_any_depth | `tools/testers/infer.py` |
| depth_pro | `src/depth_pro/depth_pro.py` |
| metric3d_v2 | `hubconf.py` |
| moge_2 | `moge/model/v2.py` — `infer()` |
| metric_anything | `models/student_pointmap/infer.py` (MoGe 위임) |
| unidepth_v2 | 저장소 README |
| unik3d | `unik3d/models/unik3d.py` — `infer()` |
| vggt | `vggt/utils/load_fn.py` |
| streamvggt | `src/streamvggt/utils/load_fn.py` |

---

## 10. 요약 — 통일 축과 모델별 축

**통일하는 것은 "유형과 절차"이고, 모델별로 남기는 것은 "무엇을 쓰는지"다.**

| 축 | 공통 골격 | 모델별 선언 (spec.json) |
| --- | --- | --- |
| 셋업 | 클론·rename·환경생성·requirements·가중치 다운로드 절차 | URL/repo id, 경로, 환경 이름, method |
| 전처리 | 리사이즈 6유형 + 정규화 4종 구현 | 어느 유형·어느 파라미터 |
| 입출력 | 버퍼 할당·바인딩·실행 | 이름·rank·개수·dtype·해상도 |
| 후처리 | 언패딩/crop/intrinsics/mask 구현 | 어느 것을 쓰는지, 필요한 상태 |
| 벤치마크 | 워밍업·반복·타이밍·JSON 스키마 | reference precision |
| 정확도 | 지표 계산기 | 출력 종류별 어느 지표 |
| 시각화 | 컬러맵·저장 | 출력 종류 |

12개 모델이 전부 다르지만, **다른 방식이 6종·5종·4종으로 수렴한다.**
따라서 "전부 같게"도 "전부 따로"도 아닌, **유형 구현 + 선언**이 맞는 구조다.
