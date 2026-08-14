# 모델 후보 검토 기록

## 1. `later/` 백업 16개 (2026-08-14)

당시 `later/`에 보관하던 16개 백업을 보고 **현재 14모델
TensorRT 비교 체계에 무엇을 추가할 가치가 있는지** 검토했다. 직접 추가할 후보가
없다는 결론 뒤 소스 백업과 자동 생성 캐시는 저장소에서 삭제했다. 삭제 전 파일은
Git 커밋 `d5dc9cf`에서 복구할 수 있다.

이 문서는 정적 검토 결과다. 로컬 파일, 공식 저장소 README, 공개 체크포인트와
라이선스를 대조했지만 후보 모델을 실제로 실행하거나 ONNX/TensorRT로 검증하지는
않았다. 따라서 아래의 `난이도`는 측정값이 아니라 코드 구조에 근거한 판단이며,
속도·VRAM·정확도는 아직 모른다.

### 먼저 구분할 것

현재 체계는 기본적으로 **한 RGB 이미지 → 깊이/점맵** 계약과 DIODE 단일 이미지
평가를 사용한다. 삭제한 백업에는 이 계약과 다른 프로젝트가 섞여 있었다.

| 종류 | 후보 | 현재 14모델 표에 바로 추가 가능한가 |
| --- | --- | --- |
| 단일 이미지 깊이 | BRIDGE, SIDepth | 가능성은 있으나 새 변환 구현 필요 |
| RGB + 외부 depth prior | Prior Depth Anything | 별도 입력·평가 계약이 필요 |
| 비디오 깊이 | FlashDepth, Video Depth Anything, DKT | 불가. temporal state와 비디오 평가가 필요 |
| 다중 프레임 3D | Align3R, LiteVGGT, MapAnything, STream3R | 불가. pose/point map/sequence 계약이 필요 |
| 전체 파이프라인 | MegaSaM, VIPE, WildGS-SLAM | 단일 TensorRT 모델이 아니라 별도 제품 범위 |
| 깊이 모델이 아님 | CoTracker3, DINOv3, GeoCalib | 모델 표가 아니라 보조 구성요소 후보 |

### 결론과 추천 순서

### 1. Prior Depth Anything — 새 범주를 열 때 가장 가치 있음

[공식 저장소](https://github.com/SpatialVision/Prior-Depth-Anything)는 RGB와 희소하거나
불완전한 metric prior를 결합해 dense metric depth를 만드는 모델이다. 기존 14개에
없는 기능이라 추가 가치가 가장 크고 코드 라이선스도 Apache-2.0이다.

다만 **15번째 일반 단안 모델로 같은 표에 넣으면 안 된다.** 정답 깊이에서 prior를
만들어 입력하면 모델이 받은 정답 정보의 양에 따라 점수가 달라진다. 다음을 먼저
고정해야 한다.

- 입력: RGB, prior depth, sparse mask, 선택적 geometric depth
- prior 생성: 점 개수·패턴·노이즈·최소/최대 깊이·난수 seed
- 평가: RGB-only 모델과 분리된 `prior-assisted` 표
- 누수 방지: 모델에 준 prior 픽셀과 평가 픽셀을 구분할지 여부

당시 `later/Prior_Depth_Anything/onnx_export.py`와 `onnx2trt.py`는 이름과 달리
Prior Depth Anything을 불러오지 않는다. 둘 다 `VGGTDepthOnlyWrapper`와
`facebook/VGGT-1B`을 사용하므로 재사용할 수 있는 변환 결과가 아니다. 통합은
업스트림 모델에서 처음부터 다시 시작해야 한다.

### 2. Video Depth Anything Small — 비디오 트랙을 열 경우 첫 후보

[공식 저장소](https://github.com/DepthAnything/Video-Depth-Anything)는 relative와
metric 모델을 각각 ViT-S/B/L로 공개했고 streaming 경로도 제공한다. Small 가중치는
Apache-2.0이고 Base/Large는 CC BY-NC 4.0이다.

당시 `later/Video_Depth_Anything/`에는 README만 있고 변환 코드는 없었다. 또한 공식
streaming 구현도 offline 대비 정확도 저하를 명시하므로 다음과 같은 별도 기준이
필요하다.

- 고정 clip 길이와 프레임 크기
- 첫 프레임 지연시간과 steady-state 지연시간 분리
- FPS뿐 아니라 temporal consistency 측정
- offline clip과 streaming cache를 별도 variant로 기록
- RTX 3080 10GB에서 Small 모델의 실제 VRAM smoke test

단일 이미지 DIODE 표에 섞지 않는다는 조건에서 비디오 확장 1순위다.

### 3. GeoCalib — 모델 수를 늘리지 않고 기존 metric 경로를 보완

[공식 저장소](https://github.com/cvg/GeoCalib)는 한 이미지에서 focal length와
gravity를 추정한다. 코드 Apache-2.0, 가중치 CC BY 4.0이다. 깊이 모델은 아니지만
현재 `demo.py`에서 Metric3D가 `--fx` 없이 metric 변환을 거부하는 문제를 보완할 수
있다.

권장 형태는 15번째 모델이 아니라 `--estimate-intrinsics geocalib` 같은 **선택적
보조 도구**다. 예측한 초점거리는 실제 calibration이 아니므로 결과에 `measured fx`
와 `estimated fx`를 반드시 구분해야 한다.

로컬 `infer.py`는 존재하지 않는 `onnx_export`를 import하므로 현재 실행 가능한
통합본은 아니다.

### 조건부 후보

| 후보 | 가치 | 지금 보류하는 이유 |
| --- | --- | --- |
| **FlashDepth** | 2K streaming video depth, Apache-2.0, Full/L/S 체크포인트 | 로컬 export wrapper는 Mamba sequence를 내부에서 시작하지만 state를 ONNX 입출력으로 내보내지 않는다. 단일 프레임 graph가 실제 streaming 모델과 같은지는 검증되지 않았다 |
| **LiteVGGT** | 기존 VGGT의 장면 수 확장·token merging 최적화, MIT | 공식 10배 주장은 1,000장 입력 조건이다. 현재 1장 depth-only 벤치에 같은 이득이 있다는 근거가 없고 Transformer Engine·동적 token merging의 TRT 지원을 먼저 확인해야 한다 |
| **MapAnything** | metric multi-view reconstruction, camera와 dense geometry를 함께 출력. Apache 가중치 변형도 공개 | 현재 계약보다 훨씬 넓고 BF16 중심이다. 로컬에는 PyTorch 예제만 있으며 10GB VRAM 적합성과 export 가능성이 미확인 |
| **DKT** | 투명 물체의 비디오 depth+normal이라는 명확한 특수 영역, Apache-2.0 | 일반 DIODE가 아니라 투명 물체·비디오 평가셋이 필요하다. 로컬 `infer/export/TRT`는 전부 DKT가 아닌 DAV2 복사본이라 처음부터 구현해야 한다 |
| **BRIDGE** | 단일 이미지 깊이라 현재 계약과 가장 가까움, Apache-2.0 | 로컬 `infer.py`는 정의되지 않은 `set_model()`을 호출하고 export/TRT 파일도 없다. 기존 DAV 계열 대비 추가 가치와 출력 단위를 먼저 검증해야 한다 |
| **SIDepth** | scale/shift-invariant depth를 별도 비교할 수 있음 | 로컬에는 README만 있고 기존 relative 모델 3개와 기능이 많이 겹친다. 공식 저장소 라이선스도 로컬 조사만으로 명확히 확정되지 않았다 |
| **Align3R** | 동적 영상에서 두 프레임 depth·point cloud·pose | CUDA RoPE 확장, RAFT, SAM2, Depth Pro/DAV2를 함께 쓰는 복합 파이프라인이며 로컬 변환 코드는 없다 |
| **STream3R** | causal sequential 3D reconstruction | metric-scale 버전이 공식 TODO이고 NTU S-Lab License 1.0이다. 단일 이미지 표와 다른 평가 체계가 필요하다 |

### 현재 범위에는 추가하지 않을 후보

| 후보 | 판정 근거 |
| --- | --- |
| **CoTracker3** | 출력이 depth가 아니라 2D point tracks와 visibility다. 필요하면 비디오 전처리/분석 도구로 별도 통합 |
| **DINOv3** | 깊이 헤드가 없는 범용 visual encoder다. 새 깊이 모델이 아니라 향후 백본 연구 대상 |
| **MegaSaM** | mono-depth 사전 계산, camera tracking, video depth optimization을 묶은 파이프라인. 한 엔진으로 비교할 대상이 아님 |
| **VIPE** | 카메라 intrinsics·motion·near-metric depth를 생성하는 전체 비디오 파이프라인. 내부적으로 DAV3/UniK3D 등 제3자 모델을 사용하므로 모델 하나를 추가하는 것과 다름 |
| **WildGS-SLAM** | 동적 물체 제거와 Gaussian map까지 포함한 SLAM 시스템. 출력·상태·평가 기준이 현재 범위를 벗어남 |

### 실제 통합을 시작할 때의 통과 조건

어느 후보를 선택해도 폴더를 `models/`로 옮기는 것이 시작점이 아니다. 먼저 아래
순서로 증거를 만든다.

1. 업스트림 PyTorch를 고정 입력으로 실행하고 입력·출력 tensor oracle 저장
2. 코드와 가중치 라이선스를 각각 기록
3. `spec.json`에 모든 입력, shape, dtype, 출력 의미와 단위 선언
4. ONNX와 PyTorch 출력 비교
5. TensorRT와 그 ONNX 출력 비교
6. 후보 계약에 맞는 정답 평가 설계
7. GPU 클럭 고정 후 3회 독립 빌드와 속도 측정
8. 기존 14개 표와 조건이 다르면 반드시 별도 표로 발표

현재 추천은 목적별로 하나다.

- **희소 센서/LiDAR prior를 활용하려면:** Prior Depth Anything
- **시간적으로 일관된 비디오 깊이가 필요하면:** Video Depth Anything Small
- **기존 metric 모델에 카메라 정보를 보완하려면:** GeoCalib을 보조 도구로 통합

목적이 정해지지 않은 상태에서 16개 중 하나를 단순히 “15번째 모델”로 넣는 것은
권장하지 않는다.

---

## 2. 최신 공개 모델 조사 (2026-08-15)

`later/` 밖에서 새로 찾은 후보다. **필터 둘을 먼저 걸었다: 코드와 가중치가 공개돼
있을 것, 그리고 RGB 한 장 외에 아무 입력도 요구하지 않을 것.** 이 두 가지가
"monocular depth estimation 모델을 추가한다"는 말의 내용이고, 나중에 거르면
코드 없는 모델과 카메라 파라미터가 필요한 모델이 목록에 남는다.

| 모델 | 판정 | 이유 |
| --- | --- | --- |
| **HyDen** (Meta, ICLR 2026) | **채택 — `models/hyden/`** | 단일 RGB 518x518. FAIR 비상업 라이선스는 사용을 제한할 뿐 전파하지 않는다 |
| **AnyDepth** | 조건부 | DySample 이 `F.grid_sample` 을 쓴다. 이 저장소의 15개 중 그 연산을 쓰는 모델이 하나도 없어 TensorRT 빌드가 검증된 적 없다. 작은 테스트가 먼저 |
| **YOLO26 Depth** | **기각 — 라이선스** | 아래 |
| **InfiniDepth** (CVPR 2026) | 기각 | RGB 전용 경로에도 MoGe-2 체크포인트가 필요하다(metric 복원용). 모델이 아니라 파이프라인. 임의 해상도 = 동적 shape, 이 저장소는 정적 shape 만 |
| **MoGe-3** | 대기 | 코드·가중치 미공개("coming soon", 날짜 없음). **후보가 아니라 뉴스다** |
| **UniDAC** (CVPR 2026) | 기각 | MIT 이지만 intrinsics·왜곡계수·위도 그리드를 입력으로 받는다. monocular 가 아니라 calibrated depth |
| **DAGE** (CVPR 2026) | 기각 | 입력이 `(B, N, 3, H, W)` — 다중 프레임 |
| DepthMaster · StableDPT | 기각 | 코드 없음 / 비디오 |

### YOLO26 Depth 를 기각한 이유 — 기술이 아니라 라이선스다

**기술적으로는 여기서 가장 붙이기 쉬운 모델이었다.** 공식 ONNX·TensorRT export 가
있고, 미터 단위 dense depth 를 내고, 5가지 크기가 공개돼 있으며, T4 에서
Depth Anything V2 Small 대비 7.7배라는 수치까지 나와 있다.

그런데 [Ultralytics 라이선스](https://www.ultralytics.com/license)가 AGPL-3.0 이고,
**학습된 가중치에도 기본 적용된다.** AGPL 은 파생물을 공개할 때 **"전체 파생 작업의
완전한 대응 소스 코드"** 공개를 요구하며, 내부 사용·API·SaaS·하드웨어 임베드에도
적용된다.

**이것이 이 표의 다른 비상업 모델들과 종류가 다른 점이다:**

| | |
| --- | --- |
| 비상업(NC) — `unidepth_v2` · `unik3d` · `streamvggt` · DAv2 가중치 | **사용**을 제한한다. 이 저장소의 스크립트는 MIT 그대로 |
| **AGPL-3.0** — YOLO26 | **전파한다.** `models/yolo26_depth/onnx_export.py` 가 `ultralytics` 를 import 하는 순간 이 저장소 전체가 AGPL 이 되어야 한다 |

**저장소의 라이선스를 바꾸는 것은 모델 추가 결정이 아니다.** Enterprise 라이선스를
사면 열리지만 그건 별개의 결정이고, 이 문서가 임의로 내릴 것이 아니다.

기술적으로 가장 쉬운 것이 법적으로 유일하게 불가능했다 — **추가 난도를 변환
비용으로만 재면 이 항목을 놓친다.**
