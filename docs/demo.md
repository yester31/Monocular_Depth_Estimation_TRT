# demo.py — one input, every model, side by side

`demo.py` draws what the models actually return for the same picture. It is the
P7 deliverable in PLAN.md §11, and the reason it needed a plan of its own is
that the obvious way to draw it is wrong.

## The rule this exists to enforce

Colour each result by its own minimum and maximum and every model produces the
same image. All that survives is the *shape* of the depth map; the metres go
away. `reports/gt.md` ranks eight of these models on exactly the metres that
would disappear — `depth_anything_v2` sits at 23.06% AbsRel and `metric3d_v2`
at 13.10%, and per-panel normalisation makes those two look identical.

So:

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

Every panel carries the model name, the input size the engine was built for,
what each output means, the unit, the precision, and a 12-character engine
fingerprint. All of it is read from `models/<name>/spec.json` and
`reports/bench/*.json` — nothing on screen is typed into this code
(execution rule 7).

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
| `<key>_depth.png` | the comparison, sectioned by output contract |
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

* `metric3d_v2` predicts through a canonical camera, so its metres need the
  focal length of the camera that took the picture. Without `--fx` it is drawn
  as a failed panel stating that, rather than dropped (execution rule 9). DIODE
  ships one: `--fx 886.81`.
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
excluded before running because it still has no evaluation adapter, and
Metric3D ran inference but rendered the documented error panel because this
invocation deliberately supplied no `--fx`. The other twelve models completed
inference and post-processing for all three inputs.
