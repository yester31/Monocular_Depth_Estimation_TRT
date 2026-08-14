# demo.py — one input, every model, side by side

`demo.py` draws what the models actually return for the same picture. It has
two views because a compact qualitative gallery and an auditable metric
comparison answer different questions.

## Two Views

| View | Command | Intended use |
| --- | --- | --- |
| Qualitative (default) | `python demo.py --live` | Compact README gallery. One `turbo` colour map, warm=near, and a separate 2–98% range for each model. This compares depth *shape*, not metres. |
| Metric audit | `python demo.py --live --view metric` | Detailed report. Metric models share one range and colourbar; relative and unknown-scale models are visibly separated. Metric3D additionally needs `--fx` or the explicitly estimated `--metric3d-fx-from-depth-pro`. |

Per-model normalization makes outputs with similar geometry easy to inspect,
but it removes absolute scale disagreement. The qualitative figure therefore
says that its colours do not compare metres. Use the metric view whenever the
question is numerical agreement or metric accuracy.

The metric view uses these rules:

| `depth_scale` in `spec.json` | how it is drawn |
| --- | --- |
| `metric` | one shared range in metres, one colourbar, `turbo`. The range is the 2–98 percentile pooled over every metric model in the figure |
| `relative` | its own range, `magma`, and the panel says `NOT aligned to anything; shape only, not distance` |
| `unknown` (`vggt`, `streamvggt`) | its own range, `bone`, in a section headed *the model does not claim one* |

`--align-relative-to MODEL` fits a scale and a shift from each relative model
to that model's inverse depth and redraws it on the metric axis. It then goes
in its own section, keeps a different frame colour, and its unit prints as
`m fitted`, because the metres are the reference's. A fitted model never moves
the shared axis: the range is pooled only over models that claim metres.

Every detailed metric panel carries the model name, engine input size, output
meaning, unit, precision, and a 12-character engine fingerprint. The compact
view keeps only the model name, scale class, normalization rule, and timing so
it remains readable at README width. Both read their facts from
`models/<name>/spec.json` and `reports/bench/*.json`.

## Three sources of output, one drawing path

```
python demo.py --synthetic                              # no GPU, no engines, no data
python demo.py --from-saved reports/demo/saved          # arrays saved earlier
python demo.py --live --save-outputs reports/demo/saved # on the desktop
```

Only `--live` needs TensorRT. Everything after inference — the `evaluate_gt`
adapters, the geometry, the shared range, the layout, the point clouds — is the
same code in all three cases, which is what makes the other two worth having.

* `--live` loads each engine named in `reports/bench/<model>_*.json` and runs it
  through `common.allocate_buffers` / `common.do_inference`, the same path
  `evaluate_gt.py` uses. It needs the engines on this machine.
* `--from-saved DIR` reads `DIR/<image>/<model>.npz` (also accepted:
  `DIR/<model>__<image>.npz`, `DIR/<model>.npz`, `.npy` for a single output).
  Keys are the output names in `spec.json`, or `out0`, `out1`, … The arrays are
  raw engine outputs, before any adapter.
* `--synthetic` builds arrays from an analytic room, shaped exactly as each
  engine emits them — a point map for `moge_2`, canonical inverse depth plus a
  field of view for `depth_pro`, canonical depth inside `metric3d_v2`'s padding.
  Every panel it produces is stamped `SYNTHETIC` and the footer says so. It
  checks the drawing, never the models.

`--save-outputs DIR` writes the raw arrays from whichever source ran, so a
desktop run can be drawn on a laptop later.

## What comes out

For each input image, in `--out` (default `reports/demo`):

| file | what it shows |
| --- | --- |
| `<key>_depth.png` | the selected depth view; compact qualitative by default, detailed shared-axis audit with `--view metric` |
| `<key>_inputs.png` | the tensor each model is actually fed, with the pad region drawn as a dashed box and the resize written under it |
| `<key>_outputs.png` | mask, normal, confidence — only outputs a model declares. Field of view and intrinsics print as values in the footer |
| `<key>_cloud.png`, `<key>_<model>.ply` | point clouds, for models that return a point map or a camera. A model returning only depth gets a stated reason, not an invented field of view |
| `demo_summary.json` | every number printed on every figure |

Depth maps are put back on the source pixel grid by the model's own adapter, so
a 1024x576 input comes out 1024x576 whether the engine ran at 518x518 or
616x1064.

## Inputs

With no `--image`, `demo.py` reads `data/eval/aspects.json` (via
`core/eval_inputs.py` if that module exists) and draws all three aspect ratios.
If the manifest is not there it falls back to `data/example.jpg` and says so —
one aspect ratio, where P7 wants three. `--image PATH` can be repeated.

## Things it will refuse to do

* `metric3d_v2` predicts through a canonical camera. The qualitative view can
  draw that canonical depth as shape and labels it `canonical`; it does not
  call the values metres. The metric view needs the focal length of the camera
  that took the picture and remains a stated failure without a focal source.
  Use `--fx 886.81` when the dataset supplies measured calibration, as DIODE
  does. `--metric3d-fx-from-depth-pro` instead converts Depth Pro's predicted
  horizontal FOV to pixels and records the result as estimated intrinsics. It
  requires both models in `--models` and must not be reported as measured
  camera calibration.
* `tr2m` has no adapter in `evaluate_gt.py` — its prompt differs per image — so
  it is listed in the footer as excluded rather than silently missing.
* Preprocessing previews for `depth_pro` need torch, which the shared runtime
  may not have. That panel says which import failed instead of disappearing.

## Checking it without a GPU

```
python -m pytest tests/test_viz.py -q
python demo.py --synthetic --out /tmp/demo          # three aspect ratios
```

`tests/test_viz.py` covers the shared-range property, the scale+shift fit,
whether `core/viz.py`'s padding geometry still agrees with the adapter that
scores `metric3d_v2`, the z-buffer in the point-cloud renderer, and a full
`--synthetic` → `--save-outputs` → `--from-saved` round trip at three aspect
ratios.

The `--live` path was executed on the RTX 3080 on 2026-08-14. It deserialised
the published engines and produced all four figures for each of the three
aspect ratios; the committed evidence is in `reports/demo/live/`. TR2M was
excluded before running because it still has no evaluation adapter. Those
older detailed figures show Metric3D's no-`--fx` error panel; the README's
qualitative figure was regenerated on 2026-08-15 and uses its canonical output
for shape only.
