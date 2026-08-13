# Unblocking the four metric models for ground-truth evaluation

Investigation only. Nothing here was run: no GPU, no engine rebuilt, no source
file edited. Every claim is traced to a file and line, or marked as inferred.

Eight models declare `"depth_scale": "metric"`. Four score immediately because
their preprocessing is a plain stretch resize and the inverse mapping is one
resize back. This note establishes what the other four need.

**Result up front.** One of the four is not blocked at all, two share a single
blocker with a dataset-specific escape, and one needs a number the repository
does not have.

| model | verdict | confidence |
| --- | --- | --- |
| `depth_pro` | **not blocked** — it is a plain stretch resize too, and metric with no per-image input | high |
| `metric3d_v2` | **unblocked** — the focal length it needs is published by DIODE: `fx = 886.81` | high |
| `unidepth_v2` | blocked at 518x518; genuine property, not a bug. Unblocked by one rebuild at 896x672 | high on cause, medium on the fix |
| `unik3d` | same, same | high on cause, medium on the fix |

The one thing this note changes about the plan: `metric3d_v2` was written up in
this repository as permanently blocked because "`real_focal_length` 를 알 수 없기
때문이다" (`docs/model_contracts.md:711`). That is true of an arbitrary image and
false of DIODE, which ships a single global calibration. See §1.

---

## 0. The rule everything is scored against

`core/gt.py:114-125`:

```python
def policy_for(depth_scale):
    return {"metric": "none", "relative": "scale_shift"}.get(depth_scale)
```

and `core/gt.py:38-42`:

```python
    if policy not in POLICIES:
        raise ValueError(f"unknown alignment policy {policy!r}; expected one of {POLICIES}")
    m = np.asarray(mask, dtype=bool)
    if policy == "none":
        return np.asarray(pred, dtype=np.float64), {}
```

A metric model is scored with **no alignment at all**. The docstring says why
(`core/gt.py:5-7`):

> **Alignment** is what you are allowed to fit before scoring. A metric model
> claims its output is in metres, so fitting anything to the ground truth would
> be marking its own homework.

`tests/test_gt.py:129-130` pins the other half of the rule:

```python
    assert policy_for("unknown") is None
    assert policy_for("canonical") is None
```

An output nobody has pinned down gets **no policy**, and therefore no score.
This matters below: `metric3d_v2` is `"canonical"` today whatever its spec says.

The scoring target is fixed for all four. Ground truth is a DIODE indoors
sample: depth `[768, 1024]` float in metres, mask `[768, 1024]` bool, from
`tools/make_eval_manifest.py:55-64`, whose units were read from the arrays
rather than taken on faith (`tools/make_eval_manifest.py:16-19`). So each model
must produce a `[768, 1024]` array of metres, in the camera frame of the
original RGB, and hand it to `metrics(pred, gt, mask, max_depth)` unaligned.

One thing to be aware of before costing any of this: **there is no evaluation
runner yet.** `core/gt.py` and `tools/make_eval_manifest.py` exist, but nothing
imports `core.gt` except `tests/test_gt.py`, and `data/eval/` has not been
generated — `data/` holds only `example.jpg`. So for all four models the
per-model work described below lands on top of writing the loop that reads a
manifest, runs an engine per image, and calls `align` then `metrics`. Each
model's `onnx2trt.py` currently does its preprocessing and post-processing
inline in `main()` around a benchmark; the pieces to lift are identified per
section.

---

## 1. `metric3d_v2` — canonical depth, and the focal length DIODE publishes

### The blocker, and why it dissolves

`models/metric3d_v2/spec.json:32`:

```json
    "output is canonical depth; metres need real_focal * scale / 1000 (D12)",
```

D12 is `docs/model_contracts.md:693-723`. Its core (`:699-702`):

```python
canonical_to_real_scale = real_focal_length * scale / 1000.0
pred_depth = pred_depth * canonical_to_real_scale   # 여기서부터 metric
```

and the reason it is absent (`docs/model_contracts.md:711-714`):

> **왜 이렇게 됐는지는 분명하다** — `real_focal_length` 를 알 수 없기 때문이다.
> `infer.py` 의 후보 목록(707.0493 → 1440 → 2890 → 3365.20)이 그 흔적이다.
> 마지막 값에는 `# from depth pro` 주석이 붙어 있다.

### Evidence: where every term comes from

The transform is dead code in the torch path,
`models/metric3d_v2/infer.py:122-132`:

```python
    #### de-canonical transform
    if 0 :
        real_focal_length = intrinsic[0]
        real_focal_length = 1440
        real_focal_length = 2890
        real_focal_length = 3365.20 # from depth pro

        print(f'f_length : {real_focal_length}, scale : {scale}, f_length * scale : {real_focal_length * scale}')
        canonical_focal_length = 1000.0 # 1000.0 is the focal length of canonical camera
        canonical_to_real_scale = real_focal_length * scale / canonical_focal_length
        pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
```

Note the four successive assignments: each overwrites the last, so the block as
written would have used `3365.20` had it ever run. It is a list of abandoned
guesses, not a computation.

In the TensorRT path the transform is not even present —
`models/metric3d_v2/onnx2trt.py:162-165`:

```python
    ###################### canonical camera space ######################
    # Still canonical here. The de-canonical transform would go on this line;
    # see the WARNING at the top of main() for why it does not.
    pred_depth = torch.clamp(pred_depth, 0, 300)
```

**`canonical_focal_length = 1000.0`** is a constant of the model, not of the
image. It is the focal length of the virtual camera every Metric3D training
image was reprojected onto (`docs/model_contracts.md:695-697`,
`models/metric3d_v2/README.md` "Note — the output is canonical depth").

**`scale`** is the keep-ratio resize factor, computed from the *original* image
dimensions at `models/metric3d_v2/onnx2trt.py:84` (identical line at
`infer.py:84`):

```python
    scale = min(input_sizes[0] / h, input_sizes[1] / w)
```

with `input_sizes = (616, 1064)` (`onnx2trt.py:81`), applied at `:85`:

```python
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
```

then centre-padded to exactly 616x1064 with `pad_info` kept for the inverse
(`onnx2trt.py:87-94`).

**`real_focal_length`** is `fx` **in pixels at the original image resolution**.
This is not a guess about upstream; it follows from the repo's own algebra.
`scale` maps an original-resolution focal to the network-input resolution, and
`1000` is defined at the network-input resolution, so the ratio
`real_focal * scale / 1000` is dimensionless only if `real_focal` is at
original resolution.

### A footgun to avoid

`models/metric3d_v2/infer.py:86-87`:

```python
    # remember to scale intrinsic, hold depth
    intrinsic = [707.0493, 707.0493, 604.0814, 180.5066] # from Metric3D/hubconf.py
```

The comment is upstream's, and upstream follows it with a line that multiplies
the whole intrinsic by `scale`. **This repo does not**; it folds `scale` into
line 131 instead. Both forms are correct and equivalent. Applying *both* —
scaling the intrinsic and keeping `* scale` — squares the factor and is a
silent 0.8x-to-0.64x error on DIODE. Pick one.

Also note `707.0493` is KITTI's focal, copied from `Metric3D/hubconf.py`'s demo.
It has nothing to do with any image this repository evaluates.

### Exact arithmetic for a 1024x768 DIODE frame

Computed here on CPU from the code above:

| quantity | value |
| --- | --- |
| `scale = min(616/768, 1064/1024)` | `0.8020833333333334` |
| resized `(w, h) = (int(1024*scale), int(768*scale))` | `(821, 616)` |
| `pad_h, pad_w` | `0, 243` |
| `pad_info` = `[0, 0, 121, 122]` | horizontal padding only |

The engine input stays 616x1064 for any aspect ratio, because the rule is
keep-ratio-into-a-box then pad. **No re-export and no rebuild is needed for
this model.** The 4:3 DIODE frame lands 616 rows exactly and pads only left and
right.

A wrong focal is a pure multiplicative error on every predicted metre. The four
abandoned candidates span 707 to 3365, a factor of 4.8 — which is why the
number has to be established, not chosen.

### DIODE publishes exactly this number

The repository's premise for leaving the transform out — "`real_focal_length` 를
알 수 없기 때문이다" (`docs/model_contracts.md:711`) — is true of an arbitrary
photograph and **false of this dataset**.

DIODE's devkit ships a global calibration in `intrinsics.txt`
(https://github.com/diode-dataset/diode-devkit/blob/master/intrinsics.txt):

```
The intrinsic parameters of the camera are:

[fx, fy, cx, cy] = [886.81, 927.06, 512, 384]

These are the parameters of the computational camera used to generate RGBD crops
from the scans as described in Section 3.2 of the paper; please note that fx and
fy are slightly different.
```

It is global rather than per-image **by construction**: DIODE frames are not raw
captures but rectified perspective crops re-rendered from a FARO Focus S350
scanner panorama by a fixed virtual camera, so every frame shares one intrinsic
matrix.

Cross-checked independently against the paper's stated 60° horizontal x 45°
vertical field of view at 1024x768 (computed here, CPU):

```
fx = (1024/2) / tan(60°/2) = 886.8100134752652   devkit: 886.81
fy = ( 768/2) / tan(45°/2) = 927.0580079512686   devkit: 927.06
```

Both reproduce the devkit to its published precision, and `cx, cy = 512, 384`
is exactly the image centre. Two derivations from unrelated statements agreeing
to six figures is as verified as this gets.

Corroborated by practice: the official UniDepth (`unidepth/datasets/diode.py`)
and iDisc (`idisc/dataloders/diode.py`) codebases both hardcode
`[[886.81, 0, 512], [0, 927.06, 384], [0, 0, 1]]` for every DIODE image. There
is no competing convention.

**So the conversion factor for DIODE is a known constant:**

| using | `real_focal * 0.8020833 / 1000` |
| --- | ---: |
| `fx = 886.81` (upstream uses `intrinsic[0]`) | **0.711296** |
| `fy = 927.06` | 0.743579 |
| `(fx+fy)/2 = 906.935` | 0.727437 |

Sanity check, to reject a wrong answer rather than to choose one: DIODE indoors
has a 6-10 m median (`tools/make_eval_manifest.py:17-19`), so the *canonical*
prediction should have a median around 8.4-14.1 before conversion. If it does
not, something upstream of the conversion is wrong.

### Steps to unblock

1. **Use `fx = 886.81`**, giving `canonical_to_real_scale = 0.711296` for every
   DIODE frame. Record the devkit URL next to the number in whatever emits the
   result — a bare `0.711296` in a script is unauditable.
   Two things not to do:
   - Do not substitute `depth_pro`'s per-image `f_px`. It would turn a
     metric3d_v2 score into a joint score of two models, and
     `docs/model_contracts.md:1151` records that depth_pro's own focal estimate
     swings 3362 -> 605 across input sizes while its depth stays put — so that
     estimate is not a calibration.
   - Do not derive the focal from the ground truth. That is alignment wearing a
     different hat, and it defeats `policy_for("metric") == "none"`.
2. Correct `models/metric3d_v2/spec.json`. Line 46 says
   `"depth_scale": "metric"` while line 14 says the output is
   `"canonical depth — NOT metres"`. The spec contradicts itself, and
   `docs/model_contracts.md:47-49` says explicitly that putting this model in
   the metric column makes the comparison table lie. The value tracks what the
   pipeline actually emits, and that is now a choice rather than a constraint:
   - if the evaluation applies the de-canonical multiply, `"metric"` is correct
     and the model is scored unaligned like the other seven;
   - if it does not, the honest value is `"canonical"`, which `policy_for` maps
     to `None` and which `tests/test_gt.py:130` already expects — the model is
     then excluded from the ranking rather than scored wrongly
     (`tests/test_gt.py:130` already expects exactly this).

   What must not survive is today's state, where the spec claims `"metric"` and
   the code emits canonical. Whichever way it goes, `spec.json:14` and `:32` and
   the caveat in `README.md` have to move with it.
3. With the focal in hand, the evaluation is: engine at 616x1064 -> reshape to
   `(1,1,616,1064)` -> unpad with `pad_info` (`onnx2trt.py:159`) -> bilinear
   resize to `(768, 1024)` (`onnx2trt.py:161`) -> multiply by
   `real_focal * scale / 1000` -> `clamp(0, 300)` -> score with `align(...,
   "none")`. The multiply is a scalar and commutes with the unpad and the
   resize; only the clamp must come after it, or it clips canonical values
   against a metre-valued bound.

### What could still go wrong

- **`fx` vs `fy` — this one is real, not hypothetical.** DIODE's devkit says so
  itself: "please note that fx and fy are slightly different". The anisotropy is
  **4.34%** (`886.81` against `927.06`), because the crops were rendered with
  horizontal and vertical FOV chosen independently (60° and 45°). Upstream
  Metric3D uses `intrinsic[0]` — i.e. `fx` — and its canonical camera has a
  single focal, so the model has no way to express a non-square pixel. Taking
  `fx` follows upstream and is the defensible choice; taking `fy` or the mean
  shifts every predicted metre by up to 4.3%. Whichever is used must be stated,
  because 4.3% is not negligible against the AbsRel figures this exercise will
  produce (the ready models sit in the 2-12% band per
  `docs/model_contracts.md:1125-1144`).
- **Truncation drift.** `int(w * scale)` gives an achieved horizontal scale of
  `821/1024 = 0.801758` against the nominal `0.802083`. The conversion must use
  the nominal `scale`, as upstream does. The 0.04% gap is negligible but the
  two are visibly different numbers and inviting to "fix".
- **No torch-vs-TRT golden exists for this model's semantics.**
  `reports/accuracy.md` compares the engine against the ONNX graph it was built
  from (0.709% at fp32), not against `model.inference()`. If
  `Metric3DExportModel.forward()` and `model.inference()` post-process
  differently, that difference is currently unmeasured.
- **Sanity, not fitting.** DIODE indoors has a 6-10 m median
  (`tools/make_eval_manifest.py:17-19`). After conversion the prediction's
  median should land in that neighbourhood. If it is out by 3x, the focal is
  wrong. Use this to reject a focal, never to choose one.

**Confidence: high.** The formula is quoted verbatim in three places in this
repository and the terms are pinned by its own algebra; the focal length is a
published primary-source value that two independent derivations agree on and two
official codebases already use. The residual uncertainty is the `fx`/`fy` choice
(bounded at 4.3%) and whether the export wrapper's post-processing matches
`model.inference()` (unmeasured), not whether the conversion can be done.

---

## 2 & 3. `unidepth_v2` and `unik3d` — the 3.1x is real, and avoidable here

These two share one blocker, one cause, and one fix. `unik3d` is quoted below;
`models/unidepth_v2/onnx2trt.py` is the same code with the same comment.

### The blocker

`models/unik3d/spec.json:36`: `"metric scale is 3.15x upstream at 518x518 (D11)"`.
`models/unidepth_v2/spec.json:40`: `"roughly 3.1x"`.

### Is it a bug in this repo's preprocessing or export, or a genuine property?

**A genuine property of running these models at 518x518.** Not an export
defect, not a coding error, and D11 was formally closed as such —
`docs/model_contracts.md:333`:

> **D11** | **결함 아님 — 의도된 트레이드오프. 한계를 명시하는 것으로 종결** |
> 기존 코드가 이미 모델의 리사이즈 규칙을 주석으로 파악해 두고 정적 엔진을 위해
> 518×518 stretch 를 택한 것이었다. [...] **코드 동작은 변경 없음**

Three independent lines of evidence:

**(a) The mechanism is measured, not inferred.** These two models infer a focal
length from the image as presented and feed it straight into depth.
`models/unidepth_v2/onnx2trt.py:72-77`:

```python
    #   fed to model      inferred fx
    #   original                2859.4
    #   518x700                  627.6
    #   518x518 (here)           551.1
    #
    # Metric depth is tied to focal length, so the depth values move with it.
```

**(b) It is specific to these two.** `docs/model_contracts.md:1125-1155` ran the
same experiment on six models. `depth_anything_v2`, `depth_pro`, `moge_2`,
`metric3d_v2` and `vggt` all stay within 1.12x; only these two move.
`docs/model_contracts.md:1163-1165` draws the distinction that matters:

> `depth_pro` 가 특히 시사적이다 — 초점거리 출력이 5.5배 달라져도 깊이 스케일은
> 유지된다. 즉 "초점거리를 추정한다"가 곧 "입력 크기에 취약하다"를 뜻하지는 않는다.
> `unik3d`/`unidepth_v2` 만 추정한 초점거리를 깊이에 그대로 반영한다.

**(c) The preprocessing code is doing exactly what it says.** The stretch is one
line, `models/unik3d/onnx2trt.py:98`:

```python
    raw_image = cv2.resize(raw_image, (input_w, input_h))
```

and the rule it replaces is quoted directly beneath it at `:103-116`, with the
reason it is not used (`:103-106`):

```python
    # What upstream unik3d.infer() does instead of the plain resize above.
    # Deliberately not used: new_H/new_W depend on the source aspect ratio, so
    # the input shape would vary per image and a static TensorRT engine cannot
    # accept that.
```

### The direction, so nobody reports the sign backwards

`core/resize_sensitivity.py:66-68`:

```python
        # best_scale returns the s that makes s*got match ref, so the factor
        # the candidate is off by is its reciprocal. 5.7 reports it that way --
        # unik3d at 518x518 reads as 3.15x too large, not 0.317x.
```

Predictions come out **3.15x too large**. Unaligned on ground truth that gives
roughly `AbsRel ≈ 2.1` and `delta1 ≈ 0`.

### It is not a constant, and must not be recorded as one

The 3.15 is one measurement on one image — `data/example.jpg`, 3024x2268, 4:3.
It decomposes (`docs/model_contracts.md:422-427`, `:439-443`):

| fed to the model | scale vs upstream | what is wrong with it |
| --- | ---: | --- |
| 896x672 — the size the model picks itself | 1.18x | only the pre-shrink outside the model |
| 518x700 | 1.75x | aspect right, pixel count too low |
| 518x518 | 3.15x | aspect stretched as well |

So the square stretch alone contributes `3.15 / 1.75 = 1.80x`, and the rest is
pixel count. Both terms are functions of the **source** image, which means the
factor changes with the dataset. It is a symptom, not a calibration constant,
and dividing it out would be alignment by another name.

### The escape, and it is a good one: DIODE is a single fixed resolution

The rule the models apply internally (`models/unik3d/onnx2trt.py:107-116` and
`docs/model_contracts.md:392-402`) is: pad to bring the aspect ratio inside
`[0.5, 2.5]`, then scale so the pixel count lands inside `[200000, 600000]`,
snapped to a multiple of 14.

Reconstructing that rule and applying it (computed here, CPU only):

| source | pixels | resize factor | raw (h, w) | snapped to 14 |
| --- | ---: | ---: | --- | --- |
| `data/example.jpg` 3024x2268 | 6,858,432 | 0.29578 | 670.82 x 894.43 | **672 x 896** |
| **DIODE 1024x768** | 786,432 | 0.87346 | 670.82 x 894.43 | **672 x 896** |

**Both land on exactly 896x672**, because both are 4:3 and both sit above the
600k upper bound — the target then depends only on the aspect ratio and the
bound, not on the source resolution. My reconstruction reproduces the 896x672
that `docs/model_contracts.md:399` records as measured for example.jpg, and
floor-to-14 would have given 882x658 instead, so the rounding mode is confirmed
by that data point. 896x672 is also a fixed point of the rule: 602,112 px is
barely over the bound and re-applying the rule returns 896x672 again.

This dissolves the objection that killed the idea before. The decision at
`docs/model_contracts.md:461-466` was:

> 896×672 가 품질로는 낫지만 **그 값은 이 샘플 이미지 전용**이다. [...] 정적
> 엔진으로 모든 종횡비를 덮으려면 **종횡비마다 엔진을 따로 구워야** 한다.

True in general. But **every image in DIODE indoors is 1024x768**, so the split
needs exactly one aspect ratio and therefore exactly one engine — and that
engine's size *is* the model's own native choice for every frame in it. The
residual 1.18x measured at 896x672 came from pre-shrinking 6.9M px down to 602k
outside the model (`docs/model_contracts.md:429-432`); on DIODE the pre-shrink
is only 786k -> 602k, so the residual should be substantially smaller. That is
a **prediction**, not a measurement.

### Steps to unblock

Two routes. They are not exclusive; route A is the honest fallback and route B
is the actual fix.

**Route A — score at 518x518, record the cause first.** No rebuild. The inverse
mapping already exists, `models/unik3d/onnx2trt.py:196-201`:

```python
        points = torch.from_numpy(trt_outputs[0].reshape(output_shape))
        points = F.interpolate(points, (h, w), mode="bilinear", align_corners=False)
        depth = points[:, -1:]
        depth = torch.clamp(depth, min=1e-3, max=1e3)
```

which is the exact inverse of a stretch: `(1,3,518,518)` -> bilinear -> `(1,3,768,1024)`
-> take the Z channel. Score that unaligned. **Write the expected ~3x into the
report before the run, not after** — an `AbsRel` near 2 with `delta1` near 0 is
otherwise indistinguishable from a broken model, and `models/unik3d/spec.json:36`
is the caveat that has to travel with the number.

Report alongside it, clearly labelled a **diagnostic and not a score**, the
least-squares scale between prediction and ground truth and the structure
correlation. `core/resize_sensitivity.py:54-81` is the method already written
down for exactly this separation. At 518x518 the scale-removed error was 5.6%
with correlation 0.72 (`docs/model_contracts.md:443`), so the diagnostic is what
shows the depth map is structurally alive while its metres are not.

**Route B — one re-export and rebuild each at 896x672.** This is what makes the
number mean something.

1. Confirm the resize rule against the upstream clones on the desktop —
   `get_paddings` / `get_resize_factor` in `unik3d/models/unik3d.py` and
   `unidepth/models/unidepthv2/unidepthv2.py`. The reconstruction above matches
   the one recorded data point, but the exact rounding is inferred and one
   `ceil` where I assumed `round` changes the answer.
2. Confirm from the manifest that every selected DIODE sample really is
   1024x768. `tools/make_eval_manifest.py:107-108` already writes `height` and
   `width` per sample, so this is a read of the JSON with no GPU involved. If
   the split is not uniform, route B collapses back to route A.
3. `input_h = 672` / `input_w = 896` in both `onnx_export.py` and
   `onnx2trt.py`. `docs/model_contracts.md:371` requires the two to agree and
   `tests/test_profiles.py` enforces it. 672 and 896 are both multiples of 14.
4. Add the profile to each `spec.json` — `tests/test_spec.py:78-103` parses
   `input_h` / `input_w` straight out of `onnx2trt.py` and fails if the spec
   disagrees.
5. Keep the 518x518 engines. They are the bench profile and the speed table
   depends on them; this adds a profile rather than replacing one.
6. The WARNING blocks at `models/unik3d/onnx2trt.py:62-86` and
   `models/unidepth_v2/onnx2trt.py:62-84` are checked by
   `tests/test_profiles.py` (`docs/model_contracts.md:482-483`). They must be
   amended, not deleted — the 518 caveat stays true of the 518 engine.

**Route C — feed the ground-truth intrinsics in, `unidepth_v2` only.** Now that
DIODE's calibration is known (§1: `fx = 886.81`, `fy = 927.06`, `cx, cy = 512, 384`),
the upstream option of conditioning the model on a real camera rather than
letting it guess becomes available in principle. UniDepthV2 supports it, and the
published DIODE numbers are reported both ways.

**This repository cannot do it today.** `models/unidepth_v2/onnx_export.py:59-67`
exports a single input:

```python
            input_names=["rgbs"],
            output_names=["pts_3d", "confidence", "intrinsics"],
```

through the upstream wrapper `UniDepthV2ONNX`
(`models/unidepth_v2/onnx_export.py:19`), and `intrinsics` is an *output*.
Whether that wrapper can be given a camera input at all is a question about
upstream's `unidepth/models/unidepthv2/export.py`, which is not in this
checkout — **unverified, and it needs reading on the desktop before anyone
plans around it.** `unik3d` has no `intrinsics` output at all
(`models/unik3d/spec.json:11-20`), so route C never applies to it.

If route C did work it would change what is being measured: a model told the
camera is a different claim from a model that infers it, and the two cannot
share a row in the comparison table.

### A diagnostic that is now available and worth capturing

DIODE's true `fx` is `886.81` at 1024x768. `unidepth_v2` *outputs* its inferred
intrinsics, and `models/unidepth_v2/onnx2trt.py:162-164` already maps them back
to source-image pixels:

```python
        intrinsics = torch.from_numpy(trt_outputs[2].reshape((1,3,3)))
        intrinsics = postprocess_intrinsics(intrinsics, resize_factors)
```

So the inferred `fx` can be compared directly against a known truth, per image,
for free. That converts the causal story in this section from an inference into
a measurement: if depth is off by a factor `k` and `fx` is off by roughly the
same `k`, the focal-length mechanism is confirmed on this dataset rather than
argued from example.jpg. It also predicts route B's outcome before the rebuild —
if `fx` at 896x672 comes back near 886.81, the metres will be close too.

Note the recorded inferred `fx` at 518x518 was `551.1` on example.jpg
(`models/unidepth_v2/onnx2trt.py:75`), against a true DIODE `fx` of `886.81` —
the same direction and rough magnitude as the depth error, which is the
consistency this diagnostic would nail down.

### What could still go wrong

- **The residual will not be zero.** Even at the model's own size,
  `docs/model_contracts.md:429-432` explains why pre-resizing outside the model
  is not the same as letting it resize: the resampling kernels differ and the
  model reads the difference as a different camera. On DIODE the gap should be
  small. It will not be nil.
- **`points[:, -1:]` is the Z channel, not ray distance.** For a perspective
  camera Z is depth, and that is what DIODE should be compared against — but
  only if DIODE ships z-depth. See the cross-cutting section at the end: the
  exposure is +22.7% at the corners, and it is **not specific to these two** —
  it applies equally to `moge_2` and `metric_anything`, which are in the
  supposedly-ready four.
- **`unik3d` has no `intrinsics` output** (`models/unik3d/spec.json:11-20`),
  so the focal-ratio correction floated at `docs/model_contracts.md:457-459` is
  available for `unidepth_v2` only. It is also a fit, and would need declaring
  as such.
- **fp16.** Both build at fp16 (`spec.json` `build_targets`), and
  `reports/accuracy.md` puts `unik3d` output 1 at 1.22% against its own ONNX —
  the largest in the table after `depth_anything_v3`. Small next to a 3x error;
  not small next to a 5% one, which is where route B should land.

**Confidence: high** that the 3.1x is a genuine property rather than a defect —
three independent lines of evidence and a formally closed defect ticket.
**Medium** on route B: the 896x672 arithmetic is solid and reproduces a
measured value, but the upstream rounding rule is reconstructed rather than
read, and the "every DIODE frame is 1024x768" premise is stated by the task and
not yet verified against the data.

---

## 4. `depth_pro` — not blocked

### Preprocessing type: plain stretch resize

`models/depth_pro/onnx2trt.py:56-74`:

```python
    transform = Compose([
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = transform(image_rgb)

    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    _, _, H, W = x.shape
    resize = H != img_size or W != img_size

    if resize:
        x = torch.nn.functional.interpolate(
            x,
            size=(img_size, img_size),
            mode=interpolation_mode,
            align_corners=False
        )
```

No padding, no aspect preservation, no keep-ratio: a single
`F.interpolate(..., size=(1536, 1536), mode="bilinear", align_corners=False)`.
`H, W` are captured at `:65`, **before** the resize, so they are the original
image's dimensions and remain available for the inverse.

This agrees with `core/preprocess.py:117-122`, whose `resize_stretch` docstring
names depth_pro as a user, and with `docs/model_contracts.md:1110`
("종횡비 무시 stretch"). It is the *same preprocessing family* as the four
models already declared ready. The only differences are the size (1536 rather
than 518) and that the resize runs in torch on the normalised float tensor
rather than in cv2 on uint8.

### The exact inverse mapping

`models/depth_pro/onnx2trt.py:118-134`:

```python
        canonical_inverse_depth = torch.from_numpy(trt_outputs[0].reshape(output_shapes))
        fov_deg = torch.from_numpy(trt_outputs[1])

        if f_px0 is None:
            f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        else :
            f_px = f_px0

        inverse_depth = canonical_inverse_depth * (W / f_px)
        f_px = f_px.squeeze()

        if resize:
            inverse_depth = torch.nn.functional.interpolate(
                inverse_depth, size=(H, W), mode=interpolation_mode, align_corners=False
            )

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
```

For DIODE, `H, W = 768, 1024`:

1. `canonical_inverse_depth` `(1,1,1536,1536)` x `(W / f_px)` — a scalar.
2. bilinear resize `(1,1,1536,1536)` -> `(1,1,768,1024)`, **in inverse-depth
   space**.
3. `clamp(1e-4, 1e4)` then reciprocal — metres, on the ground-truth grid.

Two things are easy to get wrong here and both are load-bearing. The resize
happens **before** the reciprocal: resizing depth and resizing inverse depth are
different operations, and only the second matches upstream. And the clamp bounds
inverse depth, so the resulting depth is bounded to `[1e-4, 1e4]` metres — it is
not a clamp on depth.

### Is its depth metric with no per-image input? Yes

`models/depth_pro/onnx2trt.py:47` sets `f_px0 = None`, so the branch at `:121`
takes the model's own predicted `fov_deg`. Substituting `:122` into `:126`:

```
f_px          = 0.5 * W / tan(fov/2)
W / f_px      = 2 * tan(fov/2)
inverse_depth = canonical_inverse_depth * 2 * tan(fov/2)
```

**`W` cancels exactly.** The metric conversion depends on nothing but the two
engine outputs. No focal length, no EXIF, no intrinsics, no calibration —
the contrast with `metric3d_v2`, which needs one number per image, is total.
`W` re-enters only if a caller supplies a real `f_px0`, which is the optional
path and not the one evaluated.

This is corroborated by measurement: `docs/model_contracts.md:1151` records that
depth_pro's *predicted focal* moves 3362 -> 605 across input sizes while its
*depth scale* stays at 1.02, i.e. the model compensates internally.
`docs/model_contracts.md:1130-1132` shows AbsRel 1.9-4.6% across three input
sizes, all with scale within 1.04.

### Steps to unblock

There is nothing to unblock. The procedure is: feed the DIODE RGB through
`transform` -> stretch to 1536x1536 -> engine -> the block at `:118-134`
verbatim -> `align(pred, gt, mask, "none")` -> `metrics(...)`. The existing code
already produces a `[768, 1024]` metric array; `:138-143` even hands it to
`bench.record` as `outputs={'depth': depth, 'f_px': f_px}`.

The only real work is lifting `:118-134` out of `main()` so an evaluation loop
can call it per image without the benchmark and the visualisation attached.

### What could still go wrong

- **Cost.** 1536x1536 at ~240 ms per image (`docs/model_contracts.md:1290`) —
  50 DIODE samples is minutes, not seconds, and the 8.79x pixel count
  (`docs/model_contracts.md:190`) makes it the expensive one in any sweep.
- **The 4:3 -> 1:1 stretch is upstream's own behaviour**, so it is not a defect
  here — but it does mean depth_pro is scored under a mild distortion. It
  measured 2.4% AbsRel at 518x518 versus 4.6% at 896x672
  (`docs/model_contracts.md:1130-1132`), so this is second-order.
- **fp16.** `reports/accuracy.md` gives 0.192% against its own ONNX — the
  smallest of the twelve. Not a concern.
- **`fov_deg` shape.** `:119` reshapes nothing and `:127` calls `.squeeze()`
  after the multiply. If the engine returns `fov_deg` as `(1,)` rather than a
  scalar the broadcast still works, but a batched loop should not assume it.

**Confidence: high.** The preprocessing type is a single unambiguous
`F.interpolate` call, the inverse is fifteen lines of existing code, and the
`W` cancellation is arithmetic rather than interpretation.

---

## 5. The evaluation procedure, side by side

Ground truth: `[768, 1024]` metres, `[768, 1024]` bool mask. Target: a
`[768, 1024]` array of metres in the original camera frame, scored by
`metrics(pred, gt, mask, max_depth)` after `align(pred, gt, mask, policy)`.

**Alignment: `"none"` for all four.** Every one of them declares
`"depth_scale": "metric"`, and `core/gt.py:114-125` maps that to `"none"` — no
scale, no shift, nothing fitted. The single exception is conditional: if
`metric3d_v2` ships without the de-canonical multiply its honest declaration is
`"canonical"`, which maps to `None`, meaning **not scored at all** rather than
scored with a fitted scale. There is no path by which any of the four earns a
fitted parameter.

| model | engine input | how the prediction gets back onto the 768x1024 grid | metric conversion |
| --- | --- | --- | --- |
| `depth_pro` | 1536x1536, stretch | multiply by `2·tan(fov/2)`, **bilinear resize the inverse depth** 1536x1536 -> 768x1024, then reciprocal and clamp | none needed — self-contained from `fov_deg` |
| `metric3d_v2` | 616x1064, keep-ratio + centre pad | unpad `[0,0,121,122]` -> 616x821 -> bilinear resize to 768x1024 | multiply by `fx·scale/1000 = 886.81·0.8020833/1000 = 0.711296` |
| `unidepth_v2` | 518x518, stretch | bilinear resize the point map 518x518 -> 768x1024, take Z | none applied; result is ~3x too large at this size |
| `unik3d` | 518x518, stretch | same | same |
| *(route B)* `unidepth_v2` / `unik3d` | 896x672, stretch | bilinear resize 672x896 -> 768x1024, take Z | none; the scale error is expected to largely disappear |

Three of the four resize back with a plain stretch; only `metric3d_v2` has an
unpad step first, and `core/preprocess.py:72-82` (`Geometry.unpad` /
`Geometry.to_source`) already implements exactly that pair.

The two ordering traps, restated because both are silent:

- `depth_pro` must resize **in inverse-depth space, before the reciprocal**
  (`models/depth_pro/onnx2trt.py:129-134`). Resizing depth and resizing inverse
  depth are different operations and only the second matches upstream.
- `metric3d_v2` must apply `clamp(0, 300)` **after** the de-canonical multiply,
  not before, or a metre-valued bound is applied to canonical values.

## Cross-cutting: SETTLED after this note was written

**DIODE stores z-depth. No conversion.** This section was right that the
question mattered and right that assuming would have been reckless -- it was
wrong only that the repository could not answer it. It could, from the data
already downloaded: back-project each depth map under both readings and ask
which one makes flat surfaces flat.

    patch    as z-depth     as range     ratio    patches
     24 px     0.0066%       0.0069%      1.05x    19294
     48 px     0.0087%       0.0149%      1.7x      4263
     96 px     0.0115%       0.0518%      4.5x       667
    160 px     0.0140%       0.1444%     10.3x       120

The z reading's flatness residual barely moves as the patch grows -- surface
texture. The range reading's grows with area, which is a curvature that is not
in the room. See tools/check_diode_convention.py, which keeps the test rather
than the conclusion.

The section below is left as written, because its statement of the *stakes* is
the reason the check was run at all.

---

## Cross-cutting: settle this before scoring any of the eight

**Is DIODE's depth z-depth or Euclidean range?** Every point-map model here
(`unik3d`, `unidepth_v2`, `moge_2`, `metric_anything`) reports the Z channel of
a point map. `depth_pro`, `metric3d_v2` and the depth_anything family report a
depth map. If the ground truth is range rather than z-depth, all eight are
compared against the wrong quantity.

The size of the mistake, computed here from DIODE's now-known intrinsics
(`fx, fy, cx, cy = 886.81, 927.06, 512, 384`), as `range/z = sqrt(1 + ((u-cx)/fx)² + ((v-cy)/fy)²)`:

| position in frame | `range / z` |
| --- | ---: |
| centre | 1.0000 |
| middle of the left/right edge | 1.1547 (**+15.5%**) |
| middle of the top/bottom edge | 1.0824 (+8.2%) |
| corner | 1.2267 (**+22.7%**) |

A systematic +22.7% at the corners falling to 0% at the centre is far larger
than the differences this exercise is trying to resolve — the ready models sit
at 2-12% AbsRel (`docs/model_contracts.md:1125-1144`). It would also not look
like a scale error, so no diagnostic in this repository would catch it: it is a
smooth radial bias that every model would appear to share, reading as "all our
models are worse at the edges".

There is reason to take the question seriously rather than assume: a FARO
scanner natively measures **range**, and DIODE's frames are re-rendered from a
scanner panorama by a virtual camera (devkit `intrinsics.txt`: "the
computational camera used to generate RGBD crops from the scans"). Whether that
re-rendering converted to z-depth is exactly the open question.

Not answerable from this repository. `tools/make_eval_manifest.py` verifies the
*units* are metres (`:16-19`, `:120-124`) but says nothing about the
convention. It is answerable from the devkit or from how the official UniDepth
and iDisc dataloaders consume the arrays, and it should be settled once, before
any of the eight is scored, because it silently affects all of them equally.

---

## Provenance

**Verified by reading this repository** (file:line quoted throughout): every
formula, preprocessing step, inverse mapping, spec field and defect reference.

**Computed here on CPU** (no GPU, no engine touched): the 1024x768 arithmetic in
§1, the 896x672 derivation in §2-3, the `fx`/`fy` FOV cross-check, and the
range/z-depth table. All are reproducible from the numbers given.

**Verified from primary external sources:**

- DIODE intrinsics `[fx, fy, cx, cy] = [886.81, 927.06, 512, 384]`, global for
  the whole dataset —
  https://github.com/diode-dataset/diode-devkit/blob/master/intrinsics.txt
- 1024x768 RGB, FARO Focus S350, coaxial camera with effectively zero baseline,
  crops rectified to 60°x45° FOV — DIODE paper arXiv:1908.00463 and
  https://diode-dataset.org/
- The field applies those exact intrinsics unmodified: `unidepth/datasets/diode.py`
  (lpiccinelli-eth/UniDepth) and `idisc/dataloders/diode.py` (SysCV/idisc) both
  hardcode `[[886.81, 0, 512], [0, 927.06, 384], [0, 0, 1]]`.

**Inferred, and flagged as such where it matters:**

- The exact `get_paddings` / `get_resize_factor` rounding in unik3d/unidepth_v2.
  Reconstructed from the constraints quoted in `models/unik3d/onnx2trt.py:107-116`;
  it reproduces the one measured data point (896x672 for 3024x2268) and floor
  rounding does not, but it has not been read from the upstream source.
- Whether `UniDepthV2ONNX` can accept a camera input (route C).
- Upstream `depth_pro` and `Metric3D` behaviour, where cited. **No upstream
  clone exists on this machine** — `models/*/Metric3D/`, `UniDepth/`, `UniK3D/`
  and `ml-depth-pro/` are all absent, and `.gitignore:4` confirms at least the
  last is expected to be a local clone. Everything asserted about upstream is
  therefore either quoted from this repo's own copies of it or marked inferred.

**Not established:** whether DIODE's depth is z-depth or Euclidean range.

**Not verified:** that every DIODE indoors sample really is 1024x768. The task
states it and route B depends on it; `tools/make_eval_manifest.py:107-108`
writes `height` and `width` per sample, so the generated manifest settles it
with no GPU.
