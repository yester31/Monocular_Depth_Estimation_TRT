# HyDen (MetaDepth)
- **HyDen** (ICLR 2026), Meta Reality Labs
- **[MetaDepth official GitHub](https://github.com/facebookresearch/metadepth)**
- **[HyDen-DA2-Large weights](https://huggingface.co/facebook/hyden-da2-relative-depth)** (released 2026-04-24)
- 2d image -> relative depth

## Why this one is here

**It is the first hybrid encoder in the table.** Every other ViT row runs one
path of attention; HyDen runs a CNN path and a ViT path together. That makes it
the first graph here that asks TensorRT to fuse convolutions and attention in
one engine — the two paths this repository has so far only exercised
separately, with `zipdepth` on the convolution side and thirteen transformers
on the other.

Upstream claims **"up to 10x faster 4K inference"**. It names no GPU and
publishes no numbers. This repository measures at 518x518 on an RTX 3080 with
the clock pinned; the row below neither confirms nor tests that claim, and the
claim is not repeated as if it did.

## Two things that are not settled, on purpose

| | |
| :--- | :--- |
| **The resize is not the Depth Anything V2 resize** | Upstream uses `torchvision.transforms.Resize((518,518))` on a PIL image — bilinear **with** antialias. `depth_anything_v2` here uses cv2 `INTER_LINEAR` — bilinear **without**. Reducing a 3024x2268 source by four is exactly where the two diverge most. `onnx_export.py` reproduces upstream's transform and records it to `reports/inputs/hyden.npy`; `onnx2trt.py` reads that file rather than rebuilding it. Until the two are compared on the same source, the DAv2 adapter is not reused. |
| **The output orientation is unmeasured** | Depth Anything V2's relative checkpoint emits disparity — large means near. HyDen replaces the decoder, and this repository does not inherit orientation by family resemblance: `zipdepth`'s was measured at **-0.892** correlation against true depth before it was written down. `spec.json` says `"output_form": "unmeasured"` and will keep saying it until `core.gt.orientation` has run. A wrong guess flips near and far, and the scale-and-shift fit hides it. |

## Setup

```bash
cd models/hyden
git clone https://github.com/facebookresearch/metadepth.git
mkdir -p metadepth/checkpoints
# HyDen-DA2-Large -> metadepth/checkpoints/hyden_da2_vitl.pth
#   https://huggingface.co/facebook/hyden-da2-relative-depth
```

Exported in the shared `trte` environment: upstream needs only
`torch torchvision`, which `trte` already has. No environment of its own.

```bash
python run.py export hyden     # ONNX + reports/inputs/hyden.npy
python run.py build  hyden     # engine + benchmark
```

## Encoder

`vitl`. Not a choice — the DA2 variant ships **one** checkpoint. The other two
MetaDepth releases (`hyden-mogev2-surface-normal`,
`hyden-mogev2-metric-point`, both 2026-05-21) are different heads, not smaller
encoders. This repository's rule is to use the smallest released encoder; here
the smallest is also the only one.

## TensorRT performance

<!-- BENCH:BEGIN -->
Not measured yet.
<!-- BENCH:END -->

## License

**FAIR Noncommercial Research License**, covering the code **and the weights**.

| | |
| :--- | :--- |
| Permitted | Noncommercial research, development, education, processing, analysis |
| Forbidden | Anything primarily for commercial advantage or monetary compensation |
| Derivatives | Allowed, and you keep ownership of them |
| Redistribution | Allowed, provided this agreement travels with it and the use is acknowledged in publications |

**It restricts use, not licensing.** Nothing here has to change licence to
carry this model — the conversion scripts stay MIT. What the licence limits is
what you may do with the weights and with anything you build from them.

Checked 2026-08-14 against
[the LICENSE file](https://github.com/facebookresearch/metadepth/blob/main/LICENSE).
The upstream LICENSE is always authoritative.
