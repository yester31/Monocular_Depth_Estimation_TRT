# 모델별 계약 조사 (Phase 0)

2026-07-31 / 코드 실측. 목적은 **무엇을 공통 골격으로 묶고 무엇을 모델별로 남길지** 가르는 것.
개별 특성을 없애려는 게 아니라, 특성을 명시적으로 선언 가능한 형태로 만드는 것.

> **측정 표는 이 문서에 없다.** 이 조사가 근거로 삼은 수치 표 7개는
> [`history.md` 4절](history.md#4-재현-불가능한-역사-측정) 로 옮겼다.
> `reports/` 아래 어떤 JSON 에서도 재생성되지 않고 측정 조건이 기록돼 있지
> 않아서, 발표 문서 본문에 두면 생성된 현재 수치와 같은 무게로 읽히기
> 때문이다 ([`findings.md`](findings.md) 9절). 지우지는 않았다 — 본문의
> 판정은 전부 그 표 위에 서 있다. 현재 수치는 `reports/comparison.md` 와
> 각 모델 README 의 `BENCH` 블록에 있고 전부 JSON 에서 생성된다.
>
> **본문 서술에 있던 재진술 수치는 2026-08-14 에 정리했다.** 문장 중간에
> 박혀 있던 `6.03%` · `3.15배` · `1.00~1.12` 는 전부 부록 A1·A2·A4·A5·A7 의
> 같은 값이었다. 문장에서는 빼고 부록을 가리킨다 — 한 값이 두 곳에 있으면
> 둘 중 어느 쪽이 원본인지 아무도 모른다. `0.2716 → 1.1892` 는 성격이 다르다.
> 측정이 아니라 `min(616/H, 1064/W)` 라는 산술이고, 아래 D12 에 유도가 있다.
>
> 남아 있는 숫자에는 각각 출처가 붙어 있다. 출처가 `reports/` 아래 JSON 이면
> 그 경로를, 부록이면 "재현 불가" 를 함께 적었다. 어느 것도 재측정 없이 다른
> 표로 옮겨 적을 수 없다.

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
| metric depth | depth_pro | AbsRel, RMSE |
| **canonical depth** | **metric3d_v2** | 스케일 정규화 후 AbsRel. **미터 아님 — D12** |
| depth + 부가 | depth_anything_v3 (`sky`) | depth 지표 + mask IoU |
| point map | unik3d, unidepth_v2, metric_anything, moge_2 | 점별 L2 (valid mask 적용) |
| geometry (scale 불명) | vggt, streamvggt | 스케일 정규화 후 AbsRel, 상관 |
| scalar | depth_pro (`fov_deg`), moge_2/metric_anything (`metric_scale`) | 절대·상대 오차 |
| confidence/mask | unik3d, unidepth_v2, moge_2, metric_anything | IoU |

→ **단일 `max|diff|`로는 검증 불가.** 출력 종류별 지표가 필요하다.

`metric3d_v2` 를 `depth_pro` 옆의 "metric" 칸에서 뺐다. 모델 자체는 metric 을
목표로 하지만, 이 저장소는 de-canonical 변환을 넣지 않아 출력이 canonical 이다.
같은 칸에 두면 비교표가 거짓말을 한다. 근거는 §5.5 D12.

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
| **metric3d_v2** | **616×1064** 고정 + mean/std 정규화 | 616×1064 *(bench 미적용)* | 이미 native. bench 추가 필요.<br>정규화는 ONNX 그래프에 이미 있다(D5 오진).<br>단 출력이 canonical 깊이다 — **D12** |
| moge_2 | 가변 — `num_tokens`(1200~3600) + 종횡비 | 388×518 | 동적 shape 또는 num_tokens 고정 버킷 |
| metric_anything | 가변 — MoGe 계열, 내부 결정 | 518×518 | 동적 shape 또는 버킷 |
| unidepth_v2 | 가변 — 긴 변 리사이즈 + **패딩** | 518×518 **stretch** | 동적 shape + **패딩 방식 구현** |
| unik3d | 가변 — 패딩 + 픽셀 수 bound, 14배수 | 518×518 **stretch** | 동적 shape + **패딩 방식 구현** |
| vggt | 518 — crop/pad, 14배수, 패딩 **흰색** | 518×518 (1024 경유, 패딩 검정) | **D3·D7** 수정하면 native ≈ bench |
| streamvggt | 518 — crop/pad, 1024 없음, 패딩 **흰색** | 518×518 (1024 경유, 패딩 검정) | **D1·D2·D3·D7·D8** 수정하면 native ≈ bench |

### 읽는 법 — 결함과 프로필 차이의 구분

| 구분 | 항목 |
| --- | --- |
| **결함 (고쳐야 함)** | D1·D2·D3·D6·D7·D8·D9. 업스트림과 다르게 **잘못** 구현된 것 — 전부 수정 완료 |
| **오진 (되돌림)** | D4·D5·D10. §5.5 의 "조사 방법에 대한 교훈" 참고 |
| **프로필 차이 (선택)** | 518 고정, 종횡비 처리 방식 — 비교 목적의 의도된 선택 |
| **표기 문제** | D12 — 동작은 맞는데 결과를 "metric" 이라 부르면 안 되는 경우 |
| **경계선** | `unidepth_v2`/`unik3d`의 stretch — bench 프로필에서는 정당하나, native 프로필에서는 패딩 방식이 필요 (D11, 측정 후 유지 결정) |

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
| D4 | **오진 → 되돌림**<br>**후속 = 프로필로 해결** | `/255` 는 호출부가 이미 하고 있었다(오진, 되돌림).<br>실제 결함인 종횡비 stretch([부록 A1](history.md#4-재현-불가능한-역사-측정) — 재현 불가)는 **bench/native 두 프로필로 해결** |
| D5 | **오진 → 되돌림** | 정규화는 `Metric3DExportModel.forward()` 로 **그래프에 이미 있다**.<br>추가하면 이중 적용. 되돌림 |
| D6 | **수정됨** | `f6ea99d` — export 388×518 로 통일 |
| D7 | **수정됨** | `8eaf8d5` — 패딩 흰색 |
| D8 | **수정됨** | `8eaf8d5` — D3 와 함께 제거 |
| D9 | **수정됨** | `f6ea99d` — 절대경로 |
| D10 | **철회 — 결함 아님** | README 에 wget 안내가 애초에 없었고(`unik3d` 와 혼동),<br>`from_pretrained` 와 `hf_hub_download` 는 같은 HF 저장소다.<br>README 문서 보완만 유지 |
| **D11** | **결함 아님 — 의도된 트레이드오프.<br>한계를 명시하는 것으로 종결** | 기존 코드가 이미 모델의 리사이즈 규칙을 주석으로 파악해 두고<br>정적 엔진을 위해 518×518 stretch 를 택한 것이었다.<br>측정되지 않았던 대가(metric 스케일이 어긋난다 — [부록 A2·A5](history.md#4-재현-불가능한-역사-측정), 재현 불가)를 소스와 문서에 명시.<br>**코드 동작은 변경 없음** |
| **D12** | **문서화 필요** | `metric3d_v2` 출력은 metric 이 아니라 **canonical** 깊이다.<br>de-canonical 변환이 `onnx2trt.py` 에는 아예 없고 `infer.py` 에는 `if 0:` 로 꺼져 있다 |
| **D13** | **실측으로 확정 → 수정됨** | `metric_anything` 이 518×518 stretch 후 `aspect_ratio=1.0` 으로 intrinsics 를 만들어<br>4:3 사진에 **정사각 화각**(fov_x = fov_y = 49.62°)을 부여했다.<br>`moge_2` 와 같은 388×518 로 변경 |
| D14 | **수정됨** | `depth_pro` 의 벤치마크 루프가 후처리를 포함해 측정하고 있었다.<br>다른 모델은 전부 추론만 잰다 → Phase 3 에서 정리 |
| D15 | **수정됨** | `onnx2trt.py` 3개가 쓰지도 않는 업스트림 패키지를 하드 import.<br>제거 후 12개 중 10개가 순정 `trte` 에서 동작 |
| D16 | **수정됨** | 실제로 돌려보니 스크립트 6개가 실행조차 안 됐다 (아래 표) |
| **D17** | **내 오진 → 정정** | vggt/streamvggt 를 "export 불가"로 보고했으나,<br>**두 파일 상단 `### NOTICE ###` 에 손편집 지시가 이미 있었다.**<br>그걸 안 따르고 실패한 것. 손편집을 자동화(`core/export_compat`)로 대체 |
| D18 | **수정됨** | `moge_2` 가 `utils3d` 함수 2개를 옛 이름으로 호출<br>(`depth_to_points`, 그리고 메시 블록의 3개) |

### 종횡비 — bench / native 두 프로필

D4 후속과 D11 을 같은 방식으로 처리했다. 지우거나 바꾸는 대신 **둘 다 제공**한다.

| 프로필 | 크기 | 성격 |
| --- | --- | --- |
| `bench` | **518×518** (전 모델 공통) | 속도 비교용. 4:3 원본을 정사각형으로 늘림 |
| `native` | **518×700** (4:3 기준) | 업스트림 방식. 종횡비 보존 |

#### 실측 — depth_anything_ac / vits / RTX 3080 / PyTorch

> **표는 부록으로 옮겼다 → [`docs/history.md` 부록 A1](history.md#4-재현-불가능한-역사-측정).** 이 수치는 JSON 원본이 없어 재현할 수 없다. 비교에 쓰지 마라.

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

> **표는 부록으로 옮겼다 → [`docs/history.md` 부록 A2](history.md#4-재현-불가능한-역사-측정).** 이 수치는 JSON 원본이 없어 재현할 수 없다. 비교에 쓰지 마라.

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

> **표는 부록으로 옮겼다 → [`docs/history.md` 부록 A3](history.md#4-재현-불가능한-역사-측정).** 이 수치는 JSON 원본이 없어 재현할 수 없다. 비교에 쓰지 마라.

metric depth 는 초점거리에 묶여 있으므로 **깊이 값 자체가 달라진다.**

**결론**: 이 두 모델에서 `native` 는 "업스트림과 같은 결과"를 뜻하지 않는다.
진짜 native 는 **원본을 그대로 넣어 모델이 스스로 896×672 를 고르게 하는 것**인데,
그러면 입력 크기가 이미지마다 달라져 정적 엔진을 만들 수 없다.

#### 896×672 를 넣으면? — 개선되지만 0 은 아니다

> **표는 부록으로 옮겼다 → [`docs/history.md` 부록 A4](history.md#4-재현-불가능한-역사-측정).** 이 수치는 JSON 원본이 없어 재현할 수 없다. 비교에 쓰지 마라.

896×672 가 오차를 215% → **17.8%** 로 크게 줄이지만 **0 이 되지 않는다.**
모델이 원본에서 896×672 를 고를 때와, 우리가 미리 896×672 로 줄여 줄 때가 다르기 때문이다 —
전자는 6.9M 픽셀에서 한 번에 축소하고, 후자는 이미 축소된 이미지를 받는다.
축소 과정의 정보 손실이 다르고, 모델은 그 차이를 "다른 카메라"로 해석한다.

#### 오차의 정체 — 대부분 스케일이지 구조가 아니다

위 백분율은 `mean|a-b| / mean|a|` 이므로 전체가 상수배로 밀리기만 해도 커진다.
스케일을 분리해 보면:

> **표는 부록으로 옮겼다 → [`docs/history.md` 부록 A5](history.md#4-재현-불가능한-역사-측정).** 이 수치는 JSON 원본이 없어 재현할 수 없다. 비교에 쓰지 마라.

(스케일 = 최소제곱 최적 배수, 상관 = 두 깊이 맵의 Pearson 상관계수)

**215% 가 보정 후 5.6% 로 떨어진다.** 깊이 맵의 구조는 대체로 살아 있고
전체가 상수배로 이동한 것이다.

#### 결론 — 용도에 따라 다르다

| 용도 | 판정 |
| --- | --- |
| 상대적 깊이 (어디가 가깝고 먼지) | 896×672 면 상관 0.9961 — **거의 완전** |
| metric 절대값 (몇 미터인지) | 스케일이 크게 틀림 — **보정 필요**. 크기별 값은 [부록 A4·A5](history.md#4-재현-불가능한-역사-측정)(재현 불가) |

**스케일 보정은 가능하다.** 두 모델은 intrinsics 를 함께 출력하므로 원본 대비
초점거리 비율로 깊이를 나누면 된다. `postprocess_intrinsics()` 가 이미 그 방향의
코드지만 현재는 intrinsics 에만 적용하고 depth 에는 적용하지 않는다.

#### 결정 — 518×518 유지, 한계를 명시 (2026-08-12)

896×672 가 품질로는 낫지만 **그 값은 이 샘플 이미지 전용**이다. 모델은
`pixels 200k~600k` + 종횡비 유지 규칙으로 크기를 고르므로, 16:9 이미지면
1036×582, 정사각형이면 770×770 이 된다. 정적 엔진으로 모든 종횡비를 덮으려면
**종횡비마다 엔진을 따로 구워야** 한다.

기존 코드를 다시 보니 **이미 알고 내린 결정이었다** — 모델의 리사이즈 규칙이
`'''...'''` 주석으로 소스에 남아 있고, 그 옆에 518×518 stretch 가 있다.
빠져 있던 것은 코드가 아니라 **그 선택의 대가가 얼마인지**였고, 이번에 측정했다.

따라서:

| | |
| --- | --- |
| 코드 | **변경 없음.** 518×518 유지 |
| `onnx2trt.py` | 상단에 WARNING — metric depth 가 업스트림과 어긋난다는 것과 그 측정치 표 포함([부록 A5](history.md#4-재현-불가능한-역사-측정) 와 같은 측정) |
| 주석 처리된 업스트림 규칙 | **왜 안 쓰는지** 설명 추가 (가변 크기 → 정적 엔진 불가) |
| metric 값이 필요한 용도 | **PyTorch 사용** |
| 비교표 | 이 한계를 반드시 표기 |

`tests/test_profiles.py` 가 두 파일에 WARNING 이 남아 있는지, 프로필 스위치가
다시 들어오지 않았는지 검사한다.

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

3024×2268 샘플에서 **두 입력의 출력이 서로 다르다**. 얼마나 다른지는
[부록 A1](history.md#4-재현-불가능한-역사-측정) 에 있고, **재현 불가**다 —
`reports/` 아래 어떤 JSON 도 그 값을 만들지 않으므로 여기에 옮겨 적지 않는다.

`preprocess_image()` 자체는 업스트림과 줄 단위로 일치한다 — 70행만 제거하면 된다.
다만 이는 엔진 입력 크기를 518×518 → 518×700 으로 바꾸므로 **재export·재빌드가 필요**하고,
`bench` 프로필(전 모델 518 통일)과 충돌한다. → §2 프로필 논의와 함께 결정해야 함.

**측정 근거**

```
upstream (keep ratio -> 518x700)
repo     (stretch to 518x518)      rel <부록 A1>
```

두 줄이 무엇을 무엇과 비교했는지가 근거고, `rel` 값 자체는 부록 A1 한 곳에만
둔다. 같은 수를 두 곳에 적으면 나중에 한쪽만 고쳐진다 — 이 문서에서 이미
그렇게 됐다(모델 README 8곳의 TensorRT 값, [`findings.md`](findings.md) 9절).

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

### D12 — `metric3d_v2` 는 metric 이 아니라 canonical 깊이를 낸다

Metric3D 는 모든 학습 데이터를 **초점거리 1000px 의 가상 카메라**로 정규화해서
학습한다(canonical camera space). 그래서 모델 출력은 그 가상 카메라 기준이고,
실제 미터로 바꾸려면 업스트림 `hubconf.py` 가 안내하는 변환이 필요하다.

```python
canonical_to_real_scale = real_focal_length * scale / 1000.0
pred_depth = pred_depth * canonical_to_real_scale   # 여기서부터 metric
```

이 저장소는 그 줄이 없다.

| 파일 | 상태 |
| --- | --- |
| `models/metric3d_v2/infer.py` | `if 0:` 로 막혀 있고, 안에 `real_focal_length` 후보가 네 개 나열돼 있다 |
| `models/metric3d_v2/onnx2trt.py` | 변환 자체가 없다. `###### canonical camera space ######` 주석만 남았다 |

**왜 이렇게 됐는지는 분명하다** — `real_focal_length` 를 알 수 없기 때문이다.
`infer.py` 의 후보 목록(707.0493 → 1440 → 2890 → 3365.20)이 그 흔적이다.
마지막 값에는 `# from depth pro` 주석이 붙어 있다. 다른 모델의 추정치를 빌려
쓰려다 만 것이다.

실측이 이 해석을 뒷받침한다(§5.7). keep-ratio 리사이즈 팩터는 **측정값이 아니라
산술**이다 — 모델이 616×1064 상자에 종횡비를 유지해 맞추므로
`scale = min(616/H, 1064/W)` 이고, 구현은 `tools/evaluate_gt.py` 의
`_metric3d_geometry()` 다. 3024×2268 에서 `0.2716`, 518×518 에서 `1.1892`,
그 사이가 **4.4배**다. 이 세 수는 위 식에 입력 크기를 넣으면 언제든 다시 나온다.

그 4.4배 동안 **깊이 스케일은 거의 움직이지 않는다.** 얼마나 안 움직이는지는
[부록 A7](history.md#4-재현-불가능한-역사-측정) 의 `metric3d_v2` 행에 있고
**재현 불가**다. 출력이 입력 크기에 불변이라는 건 곧 **아직 실제 카메라에
묶이지 않았다**는 뜻이다.

**조치: 코드 변경 없음, 표기만 정정.** 비교표에서 `metric3d_v2` 를
`depth_pro`·`unidepth_v2` 와 같은 "metric" 칸에 넣으면 안 된다. 별도 분류가 필요하다.
초점거리를 넣고 싶으면 EXIF 나 `depth_pro` 의 추정치를 쓸 수 있지만,
그건 이 저장소의 범위 밖이다.

### D13 — `metric_anything` 과 `moge_2` 가 같은 후처리를 다른 종횡비로 돌린다

둘 다 MoGe 의 `recover_focal_shift()` → `intrinsics_from_focal_center()` →
`depth_map_to_point_map()` 를 그대로 쓴다. 그런데 먹이는 크기가 다르다.

| | 입력 | `aspect_ratio` | 원본(4:3) 대비 |
| --- | --- | ---: | --- |
| `moge_2` | 388×518 | 1.335 | **일치** |
| `metric_anything` | 518×518 | 1.000 | 정사각으로 늘어남 |

`aspect_ratio` 는 `input_image.shape[3] / shape[2]` 로, 모델이 실제로 본 이미지에서
가져오므로 **내부적으로는 모순이 없다**. 문제는 그 정사각 이미지가 실제 장면이
아니라는 것이다.

#### 실측 — metric_anything / student_pointmap / RTX 3080

D4·D5·D10 이 전부 "코드만 읽고 확정"한 단계에서 오진이었으므로, 이번에는
돌려보고 확정했다.

> **표는 부록으로 옮겼다 → [`docs/history.md` 부록 A6](history.md#4-재현-불가능한-역사-측정).** 이 수치는 JSON 원본이 없어 재현할 수 없다. 비교에 쓰지 마라.

**확정.** 원본과 388×518 은 `fx/fy = 0.75 = 1/1.333` 로 4:3 을 정확히 반영한다.
518×518 만 `fov_x == fov_y` 다 — 4:3 사진을 정사각 장면이라고 보고한 것이다.

**단, 손상 범위는 intrinsics 에 한정된다.** 깊이 자체는 오히려 518×518 이
더 가깝다(7.1% vs 12.1%). 깊이는 `points[...,2] + shift` 에 `metric_scale` 만
곱해서 나오고 intrinsics 를 거치지 않기 때문이다. intrinsics 를 쓰는 것은
`depth_map_to_point_map()` 이므로, **깨지는 것은 점군과 화각이지 깊이맵이 아니다.**
`demo_pointcloud.py` 가 그 왜곡을 물려받는다.

#### 수정 — 388×518 로 변경

D11 과 달리 이건 고정 크기로 고칠 수 있고, 저장소 안에 이미 선례가 있다.
같은 MoGe 후처리를 쓰는 `moge_2` 가 388×518 을 쓴다. 두 모델을 맞췄다.

`resize_image(mode=2)` 로 런타임에 계산하지 않고 **숫자를 못박았다.** 엔진이
정적이고 파일명에 크기가 들어가므로, 16:9 입력이 오면 만든 적 없는
291×518 엔진을 찾게 되기 때문이다. 즉 388×518 은 `moge_2` 와 똑같이
**4:3 을 가정한다.**

`onnx2trt.py` 의 `resize_image()` 는 삭제했다(정적 엔진은 한 가지 크기만 쓴다).
`infer.py` 는 PyTorch 경로라 크기를 이미지에 맞출 수 있으므로 그대로 뒀다.

### D14 — `depth_pro` 만 후처리를 측정 구간 안에 넣고 있었다

`models/depth_pro/onnx2trt.py` 의 타이밍 루프가 추론뿐 아니라 **후처리 전체**를
감싸고 있었다. 그 안에는 1536×1536 → 3024×2268 CPU `interpolate` 가 들어 있다.
다른 11개 모델은 전부 추론만 잰다.

즉 README 에 적힌 `depth_pro` 의 FPS 는 다른 모델과 **같은 것을 잰 수치가 아니다.**
Phase 3 에서 후처리를 루프 밖으로 옮겼다. 함께 정리한 것:

| 항목 | 이전 | 이후 |
| --- | --- | --- |
| warmup | 5 / 10 / 20 (파일마다 다름) | 20 |
| iteration | 20 (depth_pro) / 100 | 100 |
| 시계 | `time.time()` | `time.perf_counter()` |
| 기록 | 평균만 stdout | 전 샘플 + 환경을 JSON 으로 |

**기존 README 수치는 전부 폐기 대상이다.**

---

### D15 — `onnx2trt.py` 3개가 쓰지도 않는 업스트림 패키지를 import 했다

"ONNX 이후는 전부 같은 환경(`trte`)" 이 목표인데, 세 파일이 이를 깨고 있었다.
AST 로 확인한 결과 **셋 다 한 번도 호출되지 않는다.**

| 파일 | import | 실제 사용 |
| --- | --- | --- |
| `models/metric3d_v2/onnx2trt.py` | `unidepth.models.unidepthv2.unidepthv2` | **없음.** 애초에 다른 모델 코드다 |
| `models/unidepth_v2/onnx2trt.py` | `UniDepth.unidepth...` | `''' '''` 주석 블록 안에만 등장 |
| `models/unik3d/onnx2trt.py` | `UniK3D.unik3d.models.unik3d` | `''' '''` 주석 블록 안에만 등장 |

`Metric3D_V2` 는 특히 명백하다 — Metric3D 스크립트에 UniDepth 코드가 들어 있고,
이 파일은 cv2 로 자체 keep-ratio 리사이즈·패딩을 한다.

셋 다 **하드 import** 라서, `trte` 에 해당 패키지가 없으면 스크립트가 아예
시작조차 못 한다. 제거했다.

#### 남은 예외 — 2개

제거 후 12개 중 10개가 순정 `trte` 에서 돈다. 남은 둘은 진짜로 필요하다.

| 모델 | 필요한 것 | 이유 |
| --- | --- | --- |
| `moge_2` | `MoGe.moge.utils.geometry_torch.recover_focal_shift` | 실제 후처리. 초점거리·shift 복원 |
| `metric_anything` | 같은 함수 (자체 사본) | 동일 |

`utils3d`·`trimesh` 는 pip 패키지라 `trte` 에 넣으면 되고 문제가 아니다.

**vendoring 하지 않기로 했다.** `recover_focal_shift` 는
`geometry_numpy.solve_optimal_focal_shift` 등 MoGe 의 수치 해법 체인에 의존한다.
베껴 오면 라이선스·유지보수·정확도 위험이 한꺼번에 생긴다. 12개 중 2개면
"모델별 규칙 유지" 범위 안이다. 근본 해결은 후처리를 ONNX 그래프에 넣는 쪽이고,
그건 Phase 5 과제다.

### D17 — "export 안 된다"는 내 오진이었다 (2026-08-13)

`vggt`·`streamvggt` 를 "어느 exporter 로도 안 된다"고 보고했다. **틀렸다.**

두 `onnx_export.py` 맨 위에 이미 이렇게 적혀 있었다:

```
### NOTICE ###
# Before exporting to onnx, edit line 55 in vggt/vggt/layers/rope.py.
    # positions = torch.cartesian_prod(y_coords, x_coords)   <- 원본
    yy = y_coords.unsqueeze(1).expand(-1, x_coords.size(0))
    xx = x_coords.unsqueeze(0).expand(y_coords.size(0), -1)
    positions = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
```

`cartesian_prod` 이 TorchScript exporter 에 없다는 걸 이미 파악하고 대체 코드까지
적어둔 것이다. **나는 그 지시를 따르지 않고 돌린 뒤 모델 탓을 했다.**

더 나빴던 건 그 반응으로 dynamo 로 바꾼 것이다. 원래 겪지 않아도 될 문제를
새로 만들었다 — dynamo 는 같은 `rope.py` 의 `int(positions.max())` 에서
`.item()` 때문에 unbacked 심볼 `u0` 를 만든다. VGGT 는 원래 exporter 로 되돌렸다.

#### 조치 — 손편집을 자동화로 대체

`core/export_compat.no_cartesian_prod()` 가 export 중에만 적용한다.
업스트림 클론 손편집은:

- 다음에 스크립트를 그냥 돌리는 사람 눈에 안 보인다
- 클론에서 `git pull` 한 번이면 사라진다
- 적용됐는지 아무도 검사하지 않는다
- 실패가 **5GB 체크포인트를 이미 로드한 뒤** 나온다

#### 함정 — 같은 파일이 두 번 로드된다

첫 시도는 패치를 넣고도 똑같이 실패했다. 원인:

```
same module object : False
same class object  : False
outer file: ...\vggt\vggt\layers\rope.py
inner file: ...\vggt\vggt\layers\rope.py
```

클론 루트와 내부 패키지 **둘 다 `__init__.py` 가 없어서** 네임스페이스 조각이고,
`rope.py` 가 `vggt.layers.rope` 와 `vggt.vggt.layers.rope` 두 이름으로 각각
로드된다. export 스크립트는 한쪽을, **모델은 다른 쪽을** 쓴다.
그래서 패치가 조용히 아무 일도 안 했다.

지금은 헬퍼가 로드된 `PositionGetter` 를 전부 찾아 **객체 식별자로 중복 제거**해
모두 패치한다. `sys.path` 문제(D16)도 같은 뿌리다 — 두 조각이 병합돼야
업스트림의 절대 import 가 풀린다.

### D16 — 실제로 돌려봤더니 스크립트 3개가 아예 실행조차 안 됐다

데스크탑에서 12개 모델을 빌드하기 시작하자마자 나온 것들이다. **읽어서는
안 나왔고, 돌리니까 즉시 나왔다.**

| 파일 | 증상 | 원인 |
| --- | --- | --- |
| `models/unidepth_v2/onnx2trt.py` | `NameError: 'profile'` | **내가 만든 것.** D11 되돌릴 때 대입문만 지우고 그걸 쓰는 `print` 를 남겼다 |
| `models/streamvggt/onnx2trt.py` | `NameError: 'original_coord'` | 오타. 변수는 `original_coords`(리스트)다. **추론이 다 끝난 뒤** 마지막 줄에서 터진다 |
| `models/vggt/onnx_export.py`<br>`models/vggt/onnx_export_split.py`<br>`models/vggt/infer.py` | `No module named 'vggt.models'` | 업스트림 `models/vggt.py` 가 형제 모듈을 절대 import 한다.<br>클론 루트가 `sys.path` 에 있어야 하는데 `onnx2trt_split.py` 만 그 줄을 갖고 있었다 |
| `models/depth_anything_ac/onnx_export.py` | `FileNotFoundError` (dinov2) | `infer.py` 의 `main()` 만 `copy_checkpoints()` 를 호출했다.<br>export 는 그 부수효과 없이 `set_model()` 을 부른다 |
| `models/unik3d/onnx2trt.py` | ONNX 파싱 실패 | `Resize` 의 `antialias=1` — TensorRT 미지원 |
| `models/metric3d_v2/onnx_export.py` | `No module named 'mmcv'` | mmcv 1.x 레이아웃. Windows 휠 없음 |

`tests/test_undefined_names.py` 를 추가해 `NameError` 부류를 정적으로 잡는다.
이 스크립트들은 모델 패키지와 GPU 없이는 import 조차 안 되므로, 그냥 두면
**ONNX 를 뽑고 엔진을 몇 분 굽고 난 뒤에야** 드러난다.

만드는 과정에서 왜 이런 검사가 보통 여기서 무용한지도 드러났다 —
모든 `onnx2trt.py` 가 `from common import *` 라 이름 공간이 가려진다.
그래서 import 하지 않고 **로컬 모듈을 파싱**하고(공용 `common.py` 는
TensorRT 를 요구해서 노트북에서 import 불가), `try`/`if` 블록 안까지 내려간다.
`common_runtime.py` 가 CUDA 바인딩을 `try` 안에서 import 하기 때문에,
최상위만 훑으면 `cudart` 를 미정의로 오탐한다.

#### 환경 쪽에서 나온 것 — TensorRT 11 은 이 저장소를 빌드할 수 없다

README 의 `pip install tensorrt-cu12` 가 오늘 설치하는 것은 **11.2** 이고,
그 환경에서는 **어떤 엔진도 안 만들어진다.**

| 제거된 것 | 대체 |
| --- | --- |
| `BuilderFlag.FP16` / `INT8` / `BF16` | ONNX 그래프의 타입 (strongly typed) |
| `BuilderFlag.OBEY_PRECISION_CONSTRAINTS` | 동일 |
| `builder.platform_has_fast_fp16` | 없음 (조언용이었음) |

`10.16.1.11` 로 고정했다. strongly-typed API 로 옮기려면 export 시점에 정밀도를
그래프에 박고 **정확도 기준선을 전부 다시 잡아야** 하므로 별도 과제다.

Windows 한글 콘솔(cp949)도 걸린다 — `torch.onnx` 가 찍는 ✅ 를 인코딩하지 못해
`UnicodeEncodeError` 로 죽는다. **일을 다 한 뒤에** 죽는 게 특히 나쁘다.
`PYTHONUTF8=1` 로 해결.

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
| streamvggt | `models/streamvggt/src` ← 하위 디렉터리 |
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

## 5.7 입력 크기 민감도 — 모델별 실측

`unidepth_v2`/`unik3d` 에서 발견한 문제가 다른 모델에도 있는지 확인했다.
방법: 원본과 사전 리사이즈본을 각각 돌리고, 둘 다 원본 해상도로 되돌려 비교.

> **표는 부록으로 옮겼다 → [`docs/history.md` 부록 A7](history.md#4-재현-불가능한-역사-측정).** 이 수치는 JSON 원본이 없어 재현할 수 없다. 비교에 쓰지 마라.

### 결론 — 이건 `unidepth_v2`/`unik3d` 고유 문제다

| 모델 | 판정 |
| --- | --- |
| `depth_anything_v2` | **문제 없음.** 상대 깊이라 크기에 둔감 |
| `depth_pro` | **문제 없음.** 초점거리를 3362→605 로 다르게 추정하지만 **깊이는 유지** (스케일 1.02). 내부적으로 보정하는 것으로 보인다 |
| `moge_2` | **거의 문제 없음.** 스케일 1.05~1.08 |
| `metric3d_v2` | **문제 없음.** 스스로 616×1064 박스에 keep-ratio 로 맞추므로 입력 크기를 흡수한다. 518² 의 8.26% 는 정사각 stretch 로 인한 왜곡이지 초점거리 문제가 아니다 |
| `vggt` | **문제 없음.** 스케일 0.94~1.01 |
| `unik3d`/`unidepth_v2` | **문제 있음.** 스케일이 크게 어긋난다 |

**이 표의 판정은 바로 위 부록 A7 을 읽은 것이다.** 셀에 남은 수치도 전부 그
표의 값이고 **재현 불가**다 — 어느 것도 다른 표로 옮겨 적을 수 없다. 크기별
정확한 값이 필요하면 A7 을 봐라.

`metric3d_v2` 의 측정은 다른 것도 보여준다. keep-ratio 리사이즈 팩터가
`min(616/H, 1064/W)` 이므로 3024×2268 의 `0.2716` 에서 518×518 의 `1.1892` 로
**4.4배** 달라지는데(측정이 아니라 산술 — D12 참조) 깊이는 거의 그대로다. 그게 바로
"canonical camera space" 의 정의다 — 출력이 입력 크기에 불변인 대신,
미터로 바꾸려면 `real_focal × scale / 1000` 이 따로 필요하다. 이 저장소는
그 변환을 넣지 않는다 (아래 D12).

`depth_pro` 가 특히 시사적이다 — 초점거리 출력이 5.5배 달라져도 깊이 스케일은 유지된다.
즉 "초점거리를 추정한다"가 곧 "입력 크기에 취약하다"를 뜻하지는 않는다.
`unik3d`/`unidepth_v2` 만 추정한 초점거리를 깊이에 그대로 반영한다.

**모든 모델에서 구조 상관은 0.99 이상**(unik3d 518² 제외). 어느 모델이든
사전 리사이즈가 깊이의 *형태*를 망가뜨리지는 않는다.

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
- **`mean(|a-b|/|a|)` (= AbsRel) 은 따로 잰다.** 표의 AbsRel 열이 그것이다.
  처음에는 "`unik3d` 는 깊이에 0 이 있어서 발산한다"고 적었는데 **틀렸다** —
  실제로 재보니 정확히 0 인 픽셀은 하나도 없고 최소값이 0.2732 였다.
  안전장치로 `ref > 1e-6 & got > 1e-6` 마스크를 두고 계산한다.
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

`models/vggt/onnx_export2.py` + `onnx2trt2.py`는 실험 잔재가 아니라 **실제로 동작하는 대안 경로**다.

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

---
## 5.8 uint8 전처리 — A/B 실측 (Phase 5)

호출 측이 하던 전처리(`/255` → 정규화 → transpose)를 **그래프 안으로 옮기면
이득인가.** `core/onnx_tools.add_uint8_input` 이 내보낸 ONNX 앞에
`Cast → Div → (Sub) → (Div) → Transpose` 를 붙이고, `tools/ab_input_dtype.py` 가
두 엔진을 같은 데이터로 비교한다.

**두 그래프의 본체는 동일하다.** 재export 가 아니라 기존 그래프를 고쳐 쓰므로,
시간 차이가 preamble 외의 곳에서 올 수 없다.

전송량은 1/4 로 줄지만 그래프에는 연산이 늘어난다. TensorRT 가 이를 첫
컨볼루션에 융합한다는 보장은 없다. **총시간 하나로는 어느 쪽이 일어났는지
알 수 없어서** `common_runtime.StageTimer` 로 구간을 나눠 잰다.

### 실측 — RTX 3080, TensorRT 10.16.1.11, 100 iterations / 20 warmup

| 모델 | 입력 | total | h2d | compute | d2h | 전송량 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **depth_anything_v2** | float32 NCHW | 4.289 | 1.034 | 2.840 | 0.350 | 3.22 MB |
| 518×518 fp16 | **uint8 NHWC** | **3.401** | **0.269** | 2.747 | 0.353 | **0.80 MB** |
| | 차이 | **−20.71%** | −0.765 | −0.093 | +0.003 | |
| **vggt** | float32 NCHW | 51.973 | 1.048 | 50.469 | 0.372 | 3.22 MB |
| 518×518 fp16 | **uint8 NHWC** | 50.962 | **0.276** | 50.250 | 0.383 | **0.80 MB** |
| | 차이 | −1.95% | −0.773 | −0.219 | +0.011 | |
| **depth_pro** | float32 NCHW | 242.189 | 9.056 | 230.117 | 2.925 | 28.31 MB |
| 1536×1536 fp16 | **uint8 NHWC** | 238.393 | **2.287** | **233.105** | 2.917 | **7.08 MB** |
| | 차이 | −1.57% | −6.769 | **+2.988** | −0.008 | |

### 절감은 비율이 아니라 **고정된 밀리초**다

| 모델 | 전송량 | h2d 절감 | **MB 당** |
| --- | ---: | ---: | ---: |
| depth_anything_v2 | 3.22 MB | 0.765 ms | **0.238 ms/MB** |
| vggt | 3.22 MB | 0.773 ms | **0.240 ms/MB** |
| depth_pro | 28.31 MB | 6.769 ms | **0.239 ms/MB** |

모델이 8.8 배 차이 나는데 **MB 당 절감은 1% 안에서 같다.** 이건 모델의 성질이
아니라 PCIe 대역폭(≈4.2 GB/s)이다.

**따라서 "어느 모델이 uint8 을 좋아하는가"는 잘못된 질문이다.** 절감량은
입력 바이트로 정해지고, 비율은 **분모가 결정한다.** depth_anything_v2 와
vggt 는 같은 518×518 입력에 같은 0.77 ms 를 아꼈는데, 하나는 20.7% 이고
하나는 1.95% 다 — 기준 시간이 4.3 ms 와 52.0 ms 이기 때문이다.

### 예외는 compute — 1536² 에서는 preamble 이 공짜가 아니다

518² 두 모델은 compute 가 오히려 미세하게 줄었다(preamble 흡수).
`depth_pro` 만 **+2.99 ms 늘어** h2d 절감 6.77 ms 의 44% 를 먹었다.
7M 픽셀에 Cast·Div·Transpose 를 얹으면 그 자체가 일이고, TensorRT 가
그것까지 첫 레이어에 접지는 않았다.

### 총시간 −1.5~2% 는 "이득"이라 부를 수 없다

같은 엔진을 다른 시점에 다시 재면 **약 2% 움직인다**(depth_anything_v2
4.23 → 4.31 ms, 한 런 안의 stdev 는 0.12 ms). `vggt` −1.95%, `depth_pro`
−1.57% 는 그 폭 안이다. 반면 `h2d` 와 `compute` 는 **같은 프로세스에서
연속으로** 잰 값이라 이 흔들림을 공유하지 않는다 — 그래서 구간 분리가
판정 근거이고, 총시간은 아니다.

### 출력은 같은가 — 완전히 같지는 않다

| 모델 | 출력 | 상대차 | 상관 |
| --- | --- | ---: | ---: |
| depth_anything_v2 | depth | 0.122% | 0.99911 |
| vggt | depth | 0.027% | 0.999999 |
| depth_pro | canonical_inverse_depth | 0.044% | 0.99902 |
| depth_pro | fov_deg | 0.194% | 1.00000 |

**preamble 은 엔진 정밀도로 실행된다.** fp16 엔진에서는 정규화가 fp16 로
수행되고, 호스트에서 float32 로 하던 것과 마지막 자리가 다르다. 이미 기록된
엔진↔ONNX 오차(0.03~0.92%)와 같은 대역이므로 별도 결함은 아니지만,
**"같은 계산"이 아니라 "같은 정밀도 대역의 계산"이라는 점은 기록해 둔다.**

### 판정

| 모델 | 판정 |
| --- | --- |
| `depth_anything_v2` | **채택 가치 있음** — 20.7%, 흔들림 폭의 10배 |
| `vggt` | **이득 없음** — 절감(0.77 ms)은 실재하나 52 ms 앞에서 무의미 |
| `depth_pro` | **보류** — h2d 절감 6.77 ms 는 확실하나 compute 증가가 44% 상쇄, 총시간은 흔들림 안 |

**적용 판단은 이 표가 아니라 위의 ms/MB 로 한다:** 입력이 크고 모델이 빠를수록
이득이고, 그 둘은 대개 함께 가지 않는다. 12개 중 이 조건에 맞는 것은
518×518 에서 10 ms 미만인 `depth_anything_v2` 뿐이다.

**자동 채택하지 않는다.** 계획이 "이득이 확인된 프로필에만 적용"이라고 명시했고,
채택하면 그 모델만 엔진 안에서 전처리를 하게 되어 비교표의 행 의미가 달라진다.

---

## D19 — VGGT split 경로는 현재 업스트림에서 export 불가

`onnx_export_split.py` / `onnx2trt_split.py` 는 VGGT 를 세 엔진
(aggregator → depth_head + camera_head)으로 나눈다. **단일 엔진과의 속도
비교는 한 번도 이뤄지지 않았다.** 이유를 찾으려고 실제로 돌려봤다.

### 원인 — 업스트림 aggregator 가 `None` 을 섞어 반환한다

`vggt/models/aggregator.py`:

```python
cached_layer_indices: Tuple[int, ...] = (4, 11, 17, 23),
...
if layer_idx in self.cached_layer_indices:
    output_list.append(concat_inter)
else:
    output_list.append(None)
```

`aggregated_tokens_list` 는 **24개 중 4개만 텐서**다. 그런데 split 쪽은
전부 텐서라고 가정한다:

```python
aggregated_tokens_list = torch.stack(aggregated_tokens_list)
# TypeError: expected Tensor as element 0 in argument 0, but got NoneType
```

`[24, 1, 1, 1374, 2048]` 이라는 선언된 shape 도, `onnx2trt_split.py` 의
device-to-device 전달도 같은 가정 위에 있다. **엔진 세 개의 인터페이스를
다시 정의해야 하는 문제이지, 한 줄 고칠 문제가 아니다.**

depth_head 는 sparse 리스트를 그대로 소화하므로 단일 엔진 경로는 정상이다.

### 가는 길에 드러난 것 3가지 — 전부 "단일 경로만 고쳐졌다"

| | 단일 (`onnx_export.py`) | split (`onnx_export_split.py`) |
| --- | --- | --- |
| cartesian_prod | D17 에서 `core.export_compat` 로 자동화 | **`### NOTICE ###` 손편집 지시가 그대로 남아 있었음** |
| 가중치 | 로컬 `vggt/checkpoints/model.pt` 우선 | **`from_pretrained` → 캐시가 비면 4.7 GB 다운로드** |
| exporter | `dynamo=False` 명시 | **인자 없음 → torch 2.11 기본값(dynamo)** |

`__main__` 에서는 셋 중 둘이 주석 처리돼 있어 **그냥 돌리면 depth_head 하나만
나오고 오류 없이 끝났다.** 그래서 아무도 이게 안 된다는 걸 몰랐다.

`onnx2trt_split.py` 는 Phase 3 의 공용 루프도 못 받았다 — `time.time()` 합산,
백분위 없음, 워밍업이 호스트 복사를 건너뜀. 이대로 잰 숫자를 단일 엔진 옆에
놓는 것은 **서로 다른 자를 대는 것**이다. 지금은 `core.bench` 를 쓰고
`variant='split'` 로 기록한다.

### 판정

**보류 — 포팅 작업이지 측정 작업이 아니다.** 위 3가지와 벤치 배선은 고쳐서
커밋했고, aggregator wrapper 의 재설계만 남는다. `tests/test_profiles.py` 가
이제 `onnx_export*.py` 전부와 파일 안의 모든 `torch.onnx.export` 호출을
검사한다 — 기존 검사는 `onnx_export.py` 만, 그것도 **마지막 호출 하나만** 봤다.

---

# 전처리 유형과 산술 dtype

P1 이 14개 모델의 전처리 호출부를 `spec.json` 선언과 전수 대조한 결과다
(2026-08-14 대조).

| 유형 | 모델 | 비고 |
| --- | --- | --- |
| stretch (종횡비 무시, 목표 크기를 정확히 채움) | 10 | |
| keep-ratio resize + 중앙 pad | 1 — `metric3d_v2` | |
| square pad 후 resize | 2 — `vggt` · `streamvggt` | |
| keep-ratio resize, pad 없음 | 0 (빌드된 프로필 기준) | Depth-Anything 계열 코드에 실재하고 `depth_anything_ac` 의 native 프로필이 쓴다 |

## 겉보기가 같은데 결과가 다른 곳

**`stretch` + `imagenet` 으로 똑같이 선언한 모델들이 산술 dtype 에서 갈린다.**
`uint8/255.0` 은 float64 로 승격되고 `uint8.astype(f32)/255.0` 은 float32 로
남는다. 파이썬 리스트로 평균을 내면 다시 float64 로 승격된다.

| 모델 | `/255` | ImageNet |
| --- | --- | --- |
| `depth_anything_v2` · `depth_anything_v3` · `distill_any_depth` | **f64** | **f64** |
| `depth_anything_ac` | **f32** | **f64** |

`spec.json` 에는 이 차이를 적는 필드가 없다. **선언만 보고는 알 수 없다.**

## `metric3d_v2` — "더 깔끔한" 재작성이 조용히 지울 두 가지

- 리사이즈 크기가 `int()` **버림**이지 반올림이 아니다. `core/preprocess.py` 의
  `resize_pad` 는 `int(round(...))` 를 쓴다. 지금 입력에서는 둘이 같지만
  `h*scale` 이 `.5` 를 막 넘는 입력에서는 pad 가 한 픽셀 밀린다.
- pad 색이 **0-255 단위의 ImageNet 평균**이고 텐서는 0-255 raw float 로 넘어간다.
  `spec.json` 의 `"normalise": "none"` 은 맞고 **의미가 있다** — 여기서 ImageNet
  통계를 적용하면 두 번 적용된다. ONNX 그래프가 안에서 정규화한다.

## `spec.json` 이 코드보다 더 주장하는 두 곳

| 항목 | 모델 | 실제 |
| --- | --- | --- |
| `type` 이 "keep-ratio + pad" 와 "square pad + resize" 를 둘 다 `keep_ratio_pad` 로 뭉갠다 | `metric3d_v2` · `vggt` · `streamvggt` | 좌표 복원 방식이 다른 별개 함수다 |
| ~~`verified: "byte-exact ... via evaluate_gt's adapter"` 는 과장이다~~ **해결** | 13 adapters | `evaluate_gt.check_adapter` 실측상 10개는 최대 차이 0이고, `depth_anything_ac`(2.384e-07) · `unidepth_v2`(4.768e-07) · `unik3d`(4.768e-07) 는 `< 1e-4` 근사다. 각 `spec.json` 의 `adapter_check` 에 oracle·실측 차이·허용치를 구조화했다. |

두 번째 항목은 2026-08-14에 문구와 테스트를 함께 고쳤다.
`tests/test_preprocess_spec.py` 는 이제 `"byte-exact"` 부분 문자열이 아니라
구조화된 `adapter_check` 로 어댑터 존재와 판정 근거를 검사한다.

---

# `tr2m` 평가 계약

`tr2m` 은 이미지마다 텍스트 프롬프트가 필요해 다른 13개와 평가 계약이 다르다.
아직 채점하지 않았고, **이 절이 채점 방식을 미리 못박는다**
(2026-08-14 확정).

| 질문 | 답 |
| --- | --- |
| 채점 가능한가 | **가능하다.** `"depth_scale": "metric"` 이므로 `core/gt.policy_for` 가 `none` 을 준다 — 다른 8개 metric 모델처럼 적합 없이 미터로 채점 |
| 무엇이 막고 있나 | 50장 각각의 프롬프트, 그리고 `evaluate_gt.py` 가 넣을 방법이 없는 두 번째 엔진 입력 |
| 이미지 전처리는 위험한가 | **아니다 — 확인함.** `reports/inputs/tr2m.npy` 를 `max\|diff\| 0.0` 으로 재현한다 |
| 프롬프트가 답을 얼마나 움직이나 | 측정된 한 점: 다른 방을 묘사한 프롬프트에서 **평균 깊이 +2.49%**. 이미지 하나·프롬프트 한 쌍이므로 **데이터 한 점이지 한계가 아니다** |

## 계약

**1. 프롬프트 정책은 이름이 붙고 커밋되는 산출물이다.** manifest 에
`prompt_policy` 와 `prompts` (문장 그대로)를 넣고, 임베딩은 CLIP 이 있는 아무
기계에서 한 번 계산해 `data/eval/<manifest>__<policy>_text_features.npy` 로
manifest 순서대로 커밋한다.

문장이 저장소에 있으므로 누구나 임베딩을 다시 만들어 같은 점수를 얻는다.
**임베딩만으로는 재현이 아니다 — 아무도 읽을 수 없는 숫자 768개다.** 그리고
문장을 바꾸면 정책 이름이 바뀌므로, 두 정책의 두 점수가 조용히 한 숫자가 될 수 없다.

**2. 정책은 하나가 아니라 둘을 돌린다.**

| 정책 | 문장 | 무엇을 재나 |
| --- | --- | --- |
| `generic_indoor_v1` | 50장 전부에 **한** 문장: "An indoor scene photographed with a handheld camera." | 바닥. 이미지별 정보가 scale head 에 안 가므로 **아무도 안 도와줄 때의 `tr2m`** |
| `described_v1` | 이미지마다 한 문장, 이미지를 보고 써서 그대로 커밋 | 이 저장소가 도달할 수 있는 천장 |

**대표 행은 `generic_indoor_v1`** — 누구의 문장도 믿지 않고 재현할 수 있는
쪽이다. `described_v1` 은 나란히 발표하고, **둘의 차이가 보고되는 프롬프트
민감도**다. 50장에 대한 실제 측정이며 위의 2.49% 와 달리 한 점이 아니다.

캡션 모델로 문장을 생성하는 세 번째 안은 **기각했다.** 재현은 되지만 그 숫자는
`tr2m` 과 캡셔너의 합작 점수가 되고, 둘 중 어느 쪽이 움직였는지 보고서에 남지
않는다.

**3. 보고서에 들어가는 것.** `tr2m` 은 metric 모델이므로 `reports/gt.md` 의
metric 표에 `alignment = none` 으로 들어간다. `vggt`·`streamvggt` 같은 별도 표
문제는 없다 — 측정값에 맞춰 적합하는 것이 없으므로 비교가 공정하다. 다만 한
행으로는 부족하다:

- 모델 열이 `` `tr2m` (generic_indoor_v1) `` 과 `` `tr2m` (described_v1) `` 이다.
  **정책이 `alignment` 처럼 행의 정체성에 포함된다**
- `reports/gt/` 의 JSON 이 `prompt_policy` 와 `prompts` 전체를 담는다
- 두 정책의 차이를 AbsRel 과 같은 문단에서 밝힌다. 각주가 아니다
- 정책을 하나만 돌렸으면 그렇게 적고 어느 것인지 적는다. **정책 이름 없는
  `tr2m` 행 하나는 발표 가능한 숫자가 아니다**

**4. 하면 안 되는 것.**

- **점수를 본 뒤에 프롬프트를 고르지 않는다.** 문장은 실행 전에 커밋한다.
  잘 나오는 문구를 고르는 것은 단계를 늘린 테스트셋 적합이고, scale 적합과 달리
  **누가 알아챌 파라미터를 남기지 않는다**
- 정책 이름 없이 다른 metric 모델과 비교하지 않는다. 다른 행은 전부 사람의
  도움을 못 받았다
- **2.49% 를 오차 막대로 재사용하지 않는다.** 한 이미지의 값이다
