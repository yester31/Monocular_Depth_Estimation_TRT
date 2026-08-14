# Model candidate review log

This file records why a candidate was considered or rejected. A candidate is not added to the benchmark until source code, weights, license terms, an input/output contract, and a reproducible TensorRT path have all been verified.

## Archived `later/` candidates (2026-08-14)

The former `later/` backup was reviewed and removed from the working tree. The review found no candidate that was both a better fit for the current single-image contract and ready for immediate integration.

- Prior Depth Anything: interesting for a future model-family expansion, but not needed while the current DA2/DA3 measurements are being stabilized.
- Video Depth Anything Small: requires a video track and therefore belongs outside the current single-image scope.
- GeoCalib: could improve camera calibration, but would add a separate calibration dependency rather than a depth model.
- Other archived entries: missing weights, incompatible contracts, unclear licensing, or no maintained export path.

## Current investigation (2026-08-15)

| Candidate | Decision | Reason |
| --- | --- | --- |
| MoGe-3 | Wait | Code and weights were announced as forthcoming; the MoGe-2 adapter may be reusable once released. |
| AnyDepth | Conditional | Promising lightweight decoder and an Orin deployment example, but CC BY-NC-SA 4.0 and a relative-depth contract require a scope decision. |
| UniDAC | Verify later | Metric, camera-independent behavior is promising, but the exact camera/intrinsics contract and license must be checked from the release repository. |
| DAGE | Reject for now | Requires multiple frames and outputs video geometry. |
| DepthMaster | Hold | Integration code and licensing were not sufficiently documented. |
| StableDPT | Reject for now | Video/temporal input rather than the current single-image contract. |

HyDen was removed because its gated repository and unavailable weights prevented a reproducible integration. YOLO26 Depth was rejected on licensing grounds, not model quality.

## Acceptance checklist

1. Public, versioned source and checkpoint URL.
2. License permits the intended use and is recorded separately for code and weights.
3. Single-image input/output contract matches an existing evaluation category.
4. ONNX export and TensorRT build succeed without private patches.
5. Accuracy and speed are measured on the same manifest and hardware as the baseline.
