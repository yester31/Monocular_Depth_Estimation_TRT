# 저장소 이력 — 끝난 계획과 재현 불가능한 측정

**이 문서는 작업 목록이 아니다.** 현재 할 일은 [`PLAN.md`](../PLAN.md) 한 곳에만
있다. 여기 있는 것은 이미 내린 결정, 그 근거, 그리고 지금은 다시 만들 수 없는
측정값이다.

세 문서를 합쳤다. 원문은 git 히스토리에 있다:
`REFACTOR_PLAN.md` · `CLEANUP_REVIEW.md` · `docs/model_contracts_historical.md`.

---

## 1. 리팩토링 계획 (2026-07-31, v3) — 완료

12개 모델이 각자 복제한 코드를 공유 구현으로 모으고, 결과를 기계 판독 가능하게
만드는 것이 목표였다. Phase 0~5 는 실행됐고 Phase 6 은 열지 않았다.

| Phase | 내용 | 결과 (2026-08-14 확인) |
| --- | --- | --- |
| 0 | 사실 조사 — 모델별 입출력·전처리·가중치 계약 | `docs/model_contracts.md`. **이게 없으면 나머지 Phase 의 설계 근거가 없다** 는 전제였고 실제로 그랬다 |
| 1 | 버그 수정 B1~B6, B8 | `common.py` 가 엔진에 fingerprint 를 기록(19곳). `SPARSE_WEIGHTS` 는 활성 저장소에서 제거 — `later/` 에만 남음 |
| 2 | `get_engine()` 통합 | 922줄 / 15개 파일 중복 → `core/common.py` 하나. 53곳이 이것을 쓴다 |
| 3 | 기계 판독 결과 + 벤치마크 통일 | `compare.py` 가 `reports/bench/*.json` → `reports/comparison.md`. **이 시점부터 모델 간 비교가 처음으로 성립했다** |
| 4 | manifest + 구조 재배치 | `models/*/spec.json` 14개, 소문자 폴더명 (§3) |
| 5 | uint8 전처리 선별 적용 | D2 로 결정: 변이로 채택, 기준선 유지 |
| 6 | 이후 | **열지 않았다. = D4**, `PLAN.md` §4 |

**계획과 다르게 끝난 것 하나.** Phase 4 는 루트 CLI 를 `build_engine.py` ·
`benchmark.py` · `compare.py` · `demo.py` 로 만들기로 했는데, 실제로는
`run.py` · `tune_build.py` · `compare.py` · `demo.py` 가 됐다. 같은 일을 하지만
이름이 다르다. **계획서를 고쳐 맞춘 것처럼 보이게 하지 않으려고 적어 둔다.**

### v1 에서 뒤집힌 전제

Codex 검토에서 v1 의 전제 상당수가 실제 코드와 맞지 않는 것으로 드러났다.
가장 중요한 셋:

- **518 고정은 결함이 아니라 의도된 `bench` 프로필이다.** 모델 간 속도 비교를
  위한 설계 선택이고 `native` 와 병행한다.
- **동적 shape 은 폐기.** 이전 작업에서 대부분 모델이 동적 export·빌드에 실패했다.
- **전 모델 단일 입력 계약은 불가능하다.** VGGT·StreamVGGT 는 5차원, MoGe-2 와
  TR2M 은 입력이 둘이다.

---

## 2. 저장소 정리 검토 (2026-07-31) — 대부분 반영

당시 저장소 13.4 MB, 커밋 `b2f9758` 기준.

**1번이 가장 컸다:** `get_engine()` 이 15개 파일에 922줄로 복제돼 있었고,
겉보기 변종 10개는 기능 차이가 아니라 공백·print 문구 차이였다. 시그니처도
쓰는 플래그도 전부 같았고, 유일한 실질 차이인 dynamic profile 블록은
`if dynamic_input_shapes is not None:` 로 감싸여 있어 없는 쪽과 동작이 같았다.

`core/common.py` 한 곳으로 모았고, 그때 제안대로 `builder_optimization_level`
파라미터를 넣었다. **그 파라미터로 돌린 P6 스윕이 레벨 5 가 `vggt` 를 2배
느리게 만든다는 것을 찾아냈다** ([`findings.md`](findings.md) P6).

나머지 항목의 현재 상태는 개별 확인이 필요하다. **이 목록을 체크리스트로 쓰지
마라** — "미해결" 로 보이는 것이 이미 다른 형태로 해결됐을 수 있다.

---

## 3. 디렉터리 이름 변경 — 2026-08-13

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


---

## 4. 재현 불가능한 역사 측정

**이 절의 수치는 JSON 원본이 없어 재현할 수 없다. 비교에 쓰지 마라.**

`docs/model_contracts.md` 본문에 있던 측정 표를 그대로 옮겨 왔다. 한 줄도
고치지 않았고 한 줄도 지우지 않았다.

**왜 지우지 않았나 —** 실행 규칙 9. 실패·폐기를 결과에서 삭제하지 않는다. 이
표들은 Phase 0 조사가 무엇을 보고 그런 판정을 내렸는지에 대한 유일한 기록이고,
본문의 결론은 전부 이 위에 서 있다.

**왜 본문에서 뺐나 —** 실행 규칙 7·8. `reports/` 아래 어떤 JSON 에서도 이 값이
나오지 않으므로, 발표 문서 본문에 두면 추적 가능한 수치와 구분되지 않는다.

## A1. `depth_anything_ac` — bench / native 프로필 (PyTorch)

원래 위치: 5.5 「종횡비 — bench / native 두 프로필」의 `#### 실측 — depth_anything_ac / vits / RTX 3080 / PyTorch`.

| 프로필 | 입력 | 시간 | 픽셀 | 업스트림 대비 |
| --- | --- | ---: | ---: | ---: |
| `bench` | 518×518 | **14.4 ms** | 268k | **6.029%** |
| `native` | 700×518 | 24.5 ms | 363k | **0.000%** |

## A2. `unik3d` — bench / native 프로필

원래 위치: 5.5 의 `#### 실측 — unik3d / vits / RTX 3080 (원본 해상도로 되돌려 비교)`.

| 프로필 | 시간 | depth 오차 | points 오차 | metric 스케일 |
| --- | ---: | ---: | ---: | ---: |
| `bench` 518×518 | 51.4 ms | **215.2%** | 210.8% | 3.15× |
| `native` 518×700 | 30.0 ms | **75.2%** | 81.9% | 1.75× |

## A3. `unidepth_v2` — 투입 크기별 추정 초점거리

원래 위치: 5.5 의 `#### 원인 — 이 두 모델은 스스로 리사이즈한다`.

| | fx | fy |
| --- | ---: | ---: |
| 원본 투입 (unidepth_v2) | 2859.4 | 2955.8 |
| 518×518 투입 | 551.1 | 561.4 |
| 518×700 투입 | 627.6 | 647.8 |

## A4. `unik3d` — 896×672 를 넣었을 때

원래 위치: 5.5 의 `#### 896×672 를 넣으면? — 개선되지만 0 은 아니다`.

| 넣은 크기 | 시간 | depth 오차 | metric 스케일 |
| --- | ---: | ---: | ---: |
| 원본 3024×2268 (기준) | 45.2 ms | — | 1.00× |
| **896×672** (모델이 스스로 고르는 크기) | 39.5 ms | **17.8%** | 1.18× |
| 518×700 | 29.9 ms | 75.2% | 1.75× |
| 518×518 (bench) | 27.3 ms | 215.2% | 3.15× |

## A5. `unik3d` — 스케일을 분리한 오차

원래 위치: 5.5 의 `#### 오차의 정체 — 대부분 스케일이지 구조가 아니다`.

| 입력 | 원래 오차 | 스케일 | **스케일 보정 후** | 구조 상관 |
| --- | ---: | ---: | ---: | ---: |
| 896×672 | 17.8% | 1.18× | **0.7%** | **0.9961** |
| 518×700 | 75.2% | 1.75× | **3.3%** | 0.8968 |
| 518×518 | 215.2% | 3.15× | **5.6%** | 0.7210 |

## A6. `metric_anything` — 투입 크기별 화각·초점거리·깊이

원래 위치: 5.5 D13 의 `#### 실측 — metric_anything / student_pointmap / RTX 3080`.

| 넣은 크기 | fov_x | fov_y | fx | fy | 깊이 스케일 | AbsRel | 상관 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 원본 3024×2268 | 58.37° | 45.46° | 0.8952 | 1.1936 | — | — | — |
| 388×518 | 53.37° | 41.26° | 0.9947 | 1.3280 | 1.11 | 12.1% | 0.9983 |
| **518×518 (기존)** | **49.62°** | **49.62°** | 1.0816 | 1.0816 | 1.06 | 7.1% | 0.9964 |

## A7. 입력 크기 민감도 — 6개 모델 실측

원래 위치: `## 5.7 입력 크기 민감도 — 모델별 실측`.

**RTX 3060 Laptop 아님 — RTX 3080, PyTorch, `data/example.jpg` (3024×2268, 4:3)**

| 모델 | 입력 | 스케일 | AbsRel | 스케일 보정 후 | 구조 상관 |
| --- | --- | ---: | ---: | ---: | ---: |
| **depth_anything_v2** | 896×672 | 1.00 | 2.1% | 1.13% | 0.9989 |
| (relative depth) | 700×518 | 1.00 | 19.3% | 2.39% | 0.9942 |
| | 518×518 | 0.97 | 9.8% | 3.95% | 0.9927 |
| **depth_pro** | 896×672 | 1.04 | 4.6% | 1.47% | 0.9983 |
| (metric, focal 출력) | 700×518 | 1.01 | 1.9% | 1.30% | 0.9983 |
| | 518×518 | 1.02 | 2.4% | 1.32% | 0.9980 |
| **moge_2** | 896×672 | 1.05 | 6.5% | 1.73% | 0.9989 |
| (metric, metric_scale 출력) | 700×518 | 1.08 | 10.5% | 3.63% | 0.9914 |
| | 518×518 | 1.05 | 8.3% | 3.83% | 0.9909 |
| **metric3d_v2** | 896×672 | 1.00 | 1.4% | 1.53% | 0.9983 |
| (canonical depth — 아래 참고) | 700×518 | 0.93 | 5.3% | 2.98% | 0.9944 |
| | 518×518 | 1.12 | 18.2% | 8.26% | 0.9301 |
| **vggt** | 896×672 | 1.01 | 1.8% | 1.00% | 0.9975 |
| (geometry, scale 불명) | 700×518 | 1.01 | 2.4% | 2.21% | 0.9835 |
| | 518×518 | 0.94 | 6.1% | 3.37% | 0.9649 |
| **unik3d** | 896×672 | **1.18** | 17.9% | 0.7% | 0.9961 |
| (metric, intrinsics 출력) | 700×518 | **1.75** | 75.6% | 3.3% | 0.8968 |
| | 518×518 | **3.15** | 217.5% | 5.6% | 0.7210 |

---

**[본문으로](model_contracts.md)**
