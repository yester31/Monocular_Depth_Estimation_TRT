# Model contracts

This document defines what can be compared across the 14 registered models. A model's output name is not enough: the coordinate system, scale, calibration, and postprocessing order are part of the contract.

## Output categories

| Category | Meaning | Evaluation |
| --- | --- | --- |
| Metric depth | Distance in metres after calibration | Metric errors on the ground-truth manifest |
| Relative depth | Affine-invariant ordering/shape | Scale-only or relative metrics |
| Canonical depth | A normalized camera-space representation | Do not label as metres without camera calibration |
| Point map | 3-D points in a model-defined frame | Check axes, handedness, and scale before projection |
| Geometry | 3-D structure with unknown global scale | Shape/scale-only evaluation |

## Registered contracts

| Model | Input | Output | Important exception |
| --- | ---: | --- | --- |
| Depth Anything V2 | 518×518 | Metric or relative | Checkpoint determines semantics |
| Depth Anything V3 | 518×518 | Metric plus sky mask | Preserve mask postprocessing |
| Depth Anything AC | 518×518 | Relative | No verified upstream license file |
| Distill Any Depth | 518×518 | Relative | Distilled checkpoint family |
| ZipDepth | 384×512 | Inverse relative depth | Convolutional model; not a ViT timing peer |
| Depth Pro | 1536×1536 | Metric plus focal length | High-resolution path and FOV prediction |
| Metric3D V2 | 616×1064 | Canonical depth | Requires measured or estimated `fx` for metric display |
| Metric Anything | 388×518 | Point map plus scale | Model-specific student checkpoint |
| MoGe-2 | 388×518 | Point map plus scale | Normal head is part of the exported variant |
| UniDepth V2 | 672×896 | Point map plus intrinsics | Native aspect ratio is important |
| UniK3D | 672×896 | Point map plus intrinsics | Native aspect ratio is important |
| TR2M | 434×560 | Text-conditioned metric depth | Requires an image-specific prompt |
| VGGT | 518×518 | Geometry | Global scale is unknown |
| StreamVGGT | 518×518 | Geometry | Global scale is unknown; level-4 builder exception |

## Preprocessing families

The adapters use one of these documented patterns: keep-ratio resize to a multiple of 14, keep-ratio resize with a constrained long side, no resize because the caller supplies the native shape, centered padding with unpadding, square padding with a second resize and inverse coordinate transform, or stretch resize with a stored scale factor. Normalization and channel order are model-specific and must remain in the adapter.

## Resolution policy

There is no valid universal 518×518 benchmark. Attention cost, internal resizing, and camera geometry differ by model. Native profiles are retained for models whose upstream contract depends on their original shape. Resolution changes require a new engine and a new accuracy measurement.

## Postprocessing rules

- Apply model-specific inverse resize/padding before resizing to the source image.
- Keep inverse depth versus depth explicit.
- Apply scale conversion before clipping when matching upstream Metric3D behavior.
- Do not turn canonical, relative, or scale-unknown outputs into metres without a declared calibration step.
- Preserve invalid-pixel masks and do not replace missing outputs with zeros.

## Setup contract

Each model directory contains `spec.json`, an ONNX exporter, a TensorRT builder, and a README. The spec records the model ID, environment, checkpoint path, input shape, precision, preprocessing, output semantics, and postprocessing. Export environments may import upstream packages; the TensorRT environment should contain only the runtime dependencies needed to build and measure engines.

## Known decisions

The historical audit corrected coordinate transforms, padding colors, Metric3D normalization/order, MoGe export/runtime shape mismatch, and stale upstream imports. These fixes are now encoded in the adapters and regression tests. Consult `docs/findings.md` and the model READMEs for the model-specific rationale.

## The `tr2m` evaluation contract

`tr2m` needs a text prompt per image, so its evaluation contract differs from
the other thirteen. It has not been scored yet, and **this section fixes how it
will be** — written before any number exists, because a prompt chosen after
seeing a score is a fit to the test set that leaves no parameter behind for
anyone to notice.

One measured point: a prompt describing a different room moved mean depth by
**+2.49%**. One image and one pair of prompts, so it is a data point, not a
bound.

**1. A prompt policy is a named, committed artefact.** The manifest carries
`prompt_policy` and `prompts` (the sentences verbatim). Embeddings are computed
once on any machine with CLIP and committed beside the manifest as
`data/eval/<manifest>__<policy>_text_features.npy`, in manifest order.

The sentences live in the repository so anyone can re-derive the embeddings and
get the same score. An embedding alone is not reproducible — it is 768 numbers
nobody can read. Changing a sentence changes the policy name, so two scores
under two policies can never quietly become one number.

**2. Two policies are run, not one.**

| Policy | The sentence | What it measures |
| --- | --- | --- |
| `generic_indoor_v1` | One sentence for all images: "An indoor scene photographed with a handheld camera." | The floor. No image-specific information reaches the scale head, so this is `tr2m` with nobody helping it — and it is reproducible from the repository with no extra model. |
| `described_v1` | One sentence per image, written from the image, committed verbatim | The ceiling this repository can reach. |

The headline row is `generic_indoor_v1`, because it is the one a reader can
reproduce without trusting anyone's sentences. `described_v1` is published
beside it, and **the gap between them is the reported prompt sensitivity** — a
real measurement over the whole manifest, which the 2.49% above is not.

Generating captions with a captioning model was considered and rejected: the
number would become a joint score of `tr2m` and the captioner, with nothing in
the report to say which one moved.

**3. What goes in the report.** `tr2m` is a metric model, so its rows belong in
the metric table with `alignment = none` — nothing is fitted to the
measurement, so the comparison is fair. But one row is not enough:

- the model column reads `` `tr2m` (generic_indoor_v1) `` and
  `` `tr2m` (described_v1) ``. The policy is part of the row's identity, the
  way `alignment` already is.
- the result JSON carries `prompt_policy` and the full `prompts` mapping, so
  the sentence behind any number is recoverable from the artefact alone.
- the prose states the spread between the two policies in the same paragraph as
  the AbsRel, not in a footnote.
- if only one policy is ever run, the report says so and says which. **A single
  `tr2m` row with no policy named is not a publishable number.**

**4. What must not happen.**

- **No prompt is chosen after seeing a score.** The sentences are committed
  before the run.
- **`tr2m` is not compared with the other metric models without the policy
  named.** Every other row got no help from a human; this one did.
- **The 2.49% is not reused as an error bar.** It is one image. Once both
  policies have run over the manifest, the measured spread replaces it and this
  section is rewritten.
