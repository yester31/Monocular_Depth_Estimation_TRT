# 저장소 정리 검토

검토일 2026-07-31 / 커밋 `b2f9758` 기준 / 저장소 13.4 MB

라이선스 섹션 추가는 **이미 반영**됐고(커밋은 안 함), 아래는 그 외 정리 항목입니다.
영향이 큰 순서로 정렬했습니다.

---

## 1. `get_engine()` 922줄이 15개 파일에 복제됨 — 최우선

`onnx2trt.py` 마다 `get_engine()` 을 따로 갖고 있습니다.

```
총 922줄 / 15개 파일 / 겉보기 변종 10개
```

**변종 10개는 기능 차이가 아니라 공백·print 문구 차이입니다.** 확인 결과:

- 시그니처가 전부 동일: `get_engine(onnx_file_path, engine_file_path="", precision='fp32', dynamic_input_shapes=None)`
  (`Metric_Anything` 만 따옴표 스타일이 다름)
- 쓰는 기능도 전부 동일: `FP16` / `SPARSE_WEIGHTS` / timing cache / `set_memory_pool_limit` / `profiling_verbosity`
- 유일한 실질 차이는 dynamic profile 블록 유무(7개 있음 / 8개 없음)인데, 그 블록은
  `if dynamic_input_shapes is not None:` 로 감싸여 있어 **없는 쪽과 동작이 동일**합니다
- `INT8` / `BF16` / `max_aux_streams` / `builder_optimization_level` 은 어디에서도 안 씀

**제안**: 루트 `common.py` 로 `get_engine()` 을 올리고 각 `onnx2trt.py` 는

```python
from common import get_engine
```

한 줄로 대체. **922줄 → 약 60줄**, 동작 변화 없음.

옮길 때 `builder_optimization_level` 파라미터를 같이 넣어두면 좋습니다. 큰 ViT 계열
그래프를 6GB GPU에서 빌드할 때 기본값 3으로는 실패하거나 매우 오래 걸립니다.

---

## 2. `common_runtime.py` 가 cuda-python 13 에서 깨짐 — 버그

```python
from cuda import cuda, cudart      # cuda-python 13 에서 제거된 경로
```

`pip install cuda-python` 은 현재 13.x 를 설치하므로, 새로 환경을 만든 사람은
`ImportError: cannot import name 'cuda' from 'cuda'` 로 막힙니다.
(이번 검토 중 실제로 재현했고, `cuda-python==12.6.*` 로 내려서 우회했습니다.)

**제안**: 양쪽 지원

```python
try:
    from cuda import cuda, cudart            # cuda-python < 13
except ImportError:
    from cuda.bindings import driver as cuda, runtime as cudart   # >= 13
```

루트 README 설치 안내에도 `pip install cuda-python` 한 줄이 빠져 있습니다
(`common_runtime.py` 가 필수 의존인데 문서화 안 됨).

---

## 3. README 가 참조하는 `results/` 가 저장소에 없음 — 문서 깨짐

`.gitignore` 에 `results*` 가 있어서 결과 이미지가 전부 빠집니다. 그런데
README 8곳이 `see results/example_vits_TRT.jpg` 처럼 참조합니다.

**선택지**
- (a) 대표 이미지 1~2장만 `assets/` 같은 별도 폴더에 커밋하고 README 가 그쪽을 참조
- (b) "스크립트를 돌리면 `results/` 에 생성됩니다" 로 문구 변경

(a) 가 낫습니다 — 모델별 출력 품질을 바로 비교할 수 있는 게 이 저장소의 핵심 가치입니다.

---

## 4. 잔여/중복 파일

| 파일 | 판단 |
| --- | --- |
| `Metric_Anything/infer0.py` | `infer.py` 와 병존하는 이전 버전으로 보임 |
| `VGGT/onnx_export2.py`, `VGGT/onnx2trt2.py` | 숫자 접미사 = 실험 잔재 |
| `later/` 17개 폴더, .py 17개 | WIP 보관함 |

`later/` 는 `.gitignore` 에 `# later/` 로 **주석 처리**돼 있습니다. 의도적으로 커밋 중이라면
루트 README 에 "작업 예정 목록" 이라고 한 줄 적어주는 게 좋고, 아니라면 주석을 풀면 됩니다.

`2` 접미사 파일들은 남긴다면 파일 상단에 한 줄 주석으로 `infer.py` 와 무엇이 다른지
적어두면 나중에 본인이 헷갈리지 않습니다.

---

## 5. 저장소 이름 vs 실제 내용

이름은 *Monocular Depth Estimation* 인데 optical flow 모델 4개(RAFT, NeuFlow, MeFlow,
MEMFOF)가 들어 있고, 이들은 루트 README 표에서 아예 **빠져 있었습니다**.

→ 이번에 루트 README 를 `### Depth` / `### Optical Flow` 두 표로 나눠 4개를 추가했습니다.
저장소 이름까지 바꿀지는 별개 판단입니다(이름 변경은 기존 링크가 깨짐).

---

## 6. 사소한 것

- 최근 커밋 메시지가 `backup` — 되돌릴 때 무엇이 들어있는지 알 수 없습니다
- `data/example.jpg` 2.2 MB, `video/video2.mp4` 4.2 MB → 저장소 13.4 MB 중 절반.
  지금은 괜찮지만 샘플이 늘면 Git LFS 를 고려
- `.gitignore` 에 `GEMINI.md` 가 있는데 `CLAUDE.md` / `.claude/` 는 없음

---

## 권장 커밋 분할

| # | 내용 | 위험도 |
| --- | --- | --- |
| 1 | 라이선스 섹션 추가 (**이미 작업됨**, 문서만 변경) | 없음 |
| 2 | `common_runtime.py` cuda-python 13 대응 + README 에 `cuda-python` 설치 추가 | 낮음 |
| 3 | `get_engine()` 을 `common.py` 로 통합, 15개 파일 수정 | 중간 — 모델별로 엔진 빌드 1회씩 재확인 필요 |
| 4 | `results/` 참조 정리 (대표 이미지 커밋 또는 문구 수정) | 없음 |
| 5 | 잔여 파일 정리 (`infer0.py`, `*2.py`, `later/` 방침 결정) | 낮음 |

3번은 15개 모델 전부 엔진을 다시 빌드해 확인해야 해서 시간이 가장 오래 걸립니다.
`get_engine` 이 기능적으로 동일하다는 건 확인했으므로 위험 자체는 낮습니다.
