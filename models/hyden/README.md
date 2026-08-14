# HyDen (MetaDepth)
- **HyDen** (ICLR 2026), Meta Reality Labs
- **[MetaDepth official GitHub](https://github.com/facebookresearch/metadepth)**
- **[HyDen-DA2-Large weights](https://huggingface.co/facebook/hyden-da2-relative-depth)** (released 2026-04-24)
- 2d image -> relative depth

## Why this one is here

**It is the first hybrid encoder in the table.** Every other ViT row runs one
path of attention; HyDen runs a CNN path alongside the ViT and fuses them. That
makes it the first graph here that asks TensorRT to fuse convolutions and
attention in one engine — the two paths this repository has so far only
exercised separately, with `zipdepth` on the convolution side and thirteen
transformers on the other.

It **adds to** Depth Anything's DPT head rather than replacing it: `da2/dpt.py`
builds `combine_feats` modules that concatenate the CNN and ViT features at each
stage, and `HyDenDepthAnything` is a wrapper that configures `DepthAnything`
with `add_cnn_encoder`. The decoder is the same one.

Upstream claims **"up to 10x faster 4K inference"**. It names no GPU and
publishes no numbers. This repository measures at 518x518 on an RTX 3080 with
the clock pinned; the row below neither confirms nor tests that claim, and the
claim is not repeated as if it did.

## Two things that are not settled, on purpose

| | |
| :--- | :--- |
| **The resize is not the Depth Anything V2 resize — measured** | Upstream uses `torchvision.transforms.Resize((518,518))` on a PIL image — bilinear **with** antialias. `depth_anything_v2` uses cv2 `INTER_LINEAR` — **without**. On `data/example.jpg` the two tensors differ by up to **2.52**, where the tensor's whole range is −2.12 to 2.64. Downscaling 3024×2268 to 518 aliases the gravel hard, and the two resizes land on different images, not on rounding. So `hyden` joins `depth_pro` in `core.preprocess.UNSUPPORTED`, `onnx_export.py` records upstream's own tensor to `reports/inputs/hyden.npy`, and `onnx2trt.py` refuses to build without it. |
| **The output is inverse depth, on upstream's word — not yet on ours** | `da2/dpt.py` keeps the final activation at ReLU *"to match production DepthAnything decoder behavior (always non-negative inverse depth)"*. So large means near, and `spec.json` records `inverse_depth` with that source. That is read from the code, not measured here. `zipdepth`'s orientation was confirmed at **-0.892** correlation against true depth before its row was published, and this one gets the same check with `core.gt.orientation` before it is scored — a wrong orientation flips near and far, and the scale-and-shift fit hides it. |

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

**What it does and does not reach.** There is no basis for saying this licence
turns the conversion scripts in this repository into FAIR-licensed work; they
are written here and stay MIT. But it does reach further than "use":

- Using the upstream code, the weights, or anything derived from them —
  **including a TensorRT engine built from them** — is bound by these terms.
- **Commercial use is forbidden.** Not restricted, forbidden.
- Redistributing any of it, or a derivative, requires the same terms and a copy
  of the Agreement travelling with it.

So the engine this repository builds is a derivative of the weights and carries
the licence; the script that built it does not.

Checked 2026-08-14 against
[the LICENSE file](https://github.com/facebookresearch/metadepth/blob/main/LICENSE).
The upstream LICENSE is always authoritative.
