"""One input, every model's output, side by side.

    python demo.py --synthetic                       # no GPU, no engines, no data
    python demo.py --from-saved reports/demo/saved   # from arrays somebody saved
    python demo.py --live --save-outputs reports/demo/saved   # on the desktop
    python demo.py --live --models moge_2,zipdepth --out reports/demo

The default is a compact qualitative gallery for the README: every model uses
the same colour map and near/far polarity, but its own robust range. That makes
shape easy to compare and deliberately does not compare metres. ``--view
metric`` keeps the auditable alternative: every metric model uses one shared
range and colourbar, while relative and unknown-scale outputs are separated.
See core/viz.py, where both rules live and are tested.

**Three sources of output, one drawing path.** `--live` runs the engines,
`--from-saved` reads arrays a previous run saved, and `--synthetic` makes them
up from an analytic scene so the layout can be checked on a machine with no
engines at all. Only the first needs a GPU; everything after it -- the
adapters, the geometry, the shared range, the layout -- is the same code in all
three cases, which is why the other two are worth having.

Everything on screen is read from models/<name>/spec.json and reports/bench/,
never typed here.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core import bench as bench_mod  # noqa: E402
from core import pointmap, spec as spec_mod, viz  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(ROOT, "reports", "bench")
ASPECTS = os.path.join(ROOT, "data", "eval", "aspects.json")
EXAMPLE = os.path.join(ROOT, "data", "example.jpg")
DEFAULT_OUT = os.path.join(ROOT, "reports", "demo")

SYNTHETIC_BADGE = "SYNTHETIC"


def rel(path, start=ROOT):
    """os.path.relpath that survives an output directory on another drive.

    Windows raises rather than returning an absolute path, and a demo that
    crashes after drawing every figure because --out was on D: would be a
    stupid way to lose a run.
    """
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _eval_inputs_module():
    """core/eval_inputs.py if it exists yet, else None.

    P2 is writing that module and data/eval/aspects.json in parallel with this
    file. A demo that refuses to run until another task lands is a demo nobody
    can check, so its absence is a fallback and not an error.
    """
    try:
        from core import eval_inputs        # noqa: F401
        return eval_inputs
    except Exception:
        return None


def _entries_from_module(mod):
    for fn in ("load", "load_manifest", "entries", "aspects", "load_aspects"):
        f = getattr(mod, fn, None)
        if not callable(f):
            continue
        try:
            got = f()
        except Exception:
            continue
        got = _unwrap(got)
        if got:
            return got, f"core.eval_inputs.{fn}()"
    return None, None


def _unwrap(obj):
    if isinstance(obj, dict):
        for k in ("samples", "entries", "inputs", "aspects"):
            if isinstance(obj.get(k), list):
                return obj[k]
        return None
    return obj if isinstance(obj, list) and obj else None


def resolve_inputs(explicit=None):
    """(list of input dicts, one line saying where they came from)."""
    if explicit:
        out = []
        for p in explicit:
            img = cv2.imread(p)
            if img is None:
                raise SystemExit(f"cannot read image: {p}")
            out.append({"path": p, "key": os.path.splitext(os.path.basename(p))[0],
                        "aspect": f"{img.shape[1]}:{img.shape[0]}",
                        "width": img.shape[1], "height": img.shape[0],
                        "source": "given on the command line"})
        return out, f"{len(out)} image(s) named on the command line"

    entries, where = (None, None)
    mod = _eval_inputs_module()
    if mod is not None:
        entries, where = _entries_from_module(mod)
    if entries is None and os.path.isfile(ASPECTS):
        with open(ASPECTS, encoding="utf-8") as f:
            entries = _unwrap(json.load(f))
        where = rel(ASPECTS)

    if entries:
        base = os.path.dirname(ASPECTS)
        out = []
        for e in entries:
            named = e.get("file") or e.get("rgb") or e.get("path")
            if not named:
                continue
            p = named if os.path.isabs(named) else os.path.join(base, named)
            if not os.path.isfile(p):
                continue
            out.append({
                "path": p,
                "key": os.path.splitext(os.path.basename(p))[0],
                "aspect": e.get("aspect", "?"),
                "width": e.get("width"), "height": e.get("height"),
                "source": e.get("source", ""), "license": e.get("license", ""),
                "crop": e.get("crop"), "sha256": e.get("sha256"),
            })
        if out:
            return out, f"{len(out)} aspect ratios from {where}"

    if not os.path.isfile(EXAMPLE):
        raise SystemExit(
            "no evaluation inputs and no data/example.jpg. Pass --image PATH.")
    img = cv2.imread(EXAMPLE)
    note = (f"data/example.jpg only ({img.shape[1]}x{img.shape[0]}). "
            f"data/eval/aspects.json is not here yet, so this run covers one "
            f"aspect ratio; P7 wants three. Run tools/prepare_eval_inputs.py, "
            f"or pass --image three times.")
    return [{"path": EXAMPLE, "key": "example",
             "aspect": f"{img.shape[1]}:{img.shape[0]}",
             "width": img.shape[1], "height": img.shape[0],
             "source": "data/example.jpg"}], note


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def adapters():
    """evaluate_gt's adapter table, or ({}, why).

    Imported rather than copied. Those adapters are the only checked statement
    in the repository of what each engine's raw output means -- every one is
    verified byte-for-byte against reports/inputs/<model>.npy -- and a second
    copy here would be a second thing to keep in step, which is how the
    preprocessing defects in docs/model_contracts.md happened in the first
    place. evaluate_gt imports no TensorRT at module scope, so this works on a
    machine with no GPU.
    """
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import evaluate_gt
        return dict(evaluate_gt.ADAPTERS), ""
    except Exception as e:                                    # pragma: no cover
        return {}, f"{type(e).__name__}: {e}"


def bench_runs():
    if not os.path.isdir(BENCH_DIR):
        return {}
    records = bench_mod.load_all(BENCH_DIR)
    out = {}
    for r in records:
        if r.get("variant", "single") != "single":
            continue
        # Newest measurement wins when a model has several; the record that is
        # displayed is named in the summary JSON so the choice is inspectable.
        prev = out.get(r["model"])
        if prev is None or str(r.get("timestamp", "")) > str(prev.get("timestamp", "")):
            out[r["model"]] = r
    return out


def select_models(specs, adapt, wanted=None):
    """(models to draw, {model: why not} for the rest). Nothing is dropped."""
    skipped = {}
    names = sorted(specs)
    if wanted:
        unknown = [m for m in wanted if m not in specs]
        if unknown:
            raise SystemExit(f"no spec for: {', '.join(unknown)}")
        names = [m for m in names if m in wanted]
    chosen = []
    for m in names:
        if m not in adapt:
            skipped[m] = "no adapter in evaluate_gt.py; its output contract is unverified"
            continue
        chosen.append(m)
    return chosen, skipped


# ---------------------------------------------------------------------------
# Synthetic outputs
# ---------------------------------------------------------------------------

def synthetic_scene(h, w):
    """Metres for an analytic room: a receding floor, a back wall, one blob.

    Not a model and not data. It exists so the drawing path -- adapters,
    geometry, shared range, layout -- can be exercised end to end on a machine
    that has neither engines nor DIODE. Every panel drawn from it is stamped.
    """
    ys, xs = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    floor = 1.2 / np.clip(ys - 0.02, 0.06, None)
    depth = np.minimum(floor, 7.5)
    blob = np.exp(-(((xs - 0.34) / 0.16) ** 2 + ((ys - 0.55) / 0.20) ** 2))
    depth = depth - 2.6 * blob
    depth += 0.35 * np.sin(6.0 * xs) * np.cos(4.0 * ys)
    return np.clip(depth, 0.45, 9.0)


def _model_bias(model):
    """A small deterministic scale and offset per model.

    Real models disagree about absolute distance -- that disagreement is the
    thing the shared colour range exists to show -- so a synthetic run in which
    every model returns the identical map would test the layout and hide the
    property being tested.
    """
    d = int(__import__("hashlib").sha256(model.encode()).hexdigest()[:8], 16)
    return 0.82 + (d % 41) / 100.0, ((d >> 8) % 31 - 15) / 100.0


def synthesize(model, spec, eh, ew, oh, ow, fx=None):
    """Raw outputs shaped exactly as this model's engine emits them."""
    scale, shift = _model_bias(model)
    depth_o = synthetic_scene(oh, ow) * scale + shift
    depth_e = cv2.resize(depth_o, (ew, eh), interpolation=cv2.INTER_LINEAR)
    names = [o["name"] for o in spec["outputs"]]

    if model == "depth_pro":
        fov = 55.0
        f_px = viz.focal_from_fov(fov, ow)
        canonical = (1.0 / depth_e) * (f_px / ow)
        return [canonical.astype(np.float32), np.float32([fov])]

    if model == "metric3d_v2":
        if not fx:
            fx = 1000.0
        geom = viz.input_geometry(spec, oh, ow, eh, ew)
        s = geom.inner_w / ow
        inner = cv2.resize(depth_o, (geom.inner_w, geom.inner_h),
                           interpolation=cv2.INTER_LINEAR)
        canon = inner * 1000.0 / (fx * s)
        padded = np.full((eh, ew), 0.0, np.float32)
        y0, y1, x0, x1 = geom.crop_box()
        padded[y0:y1, x0:x1] = canon
        return [padded]

    if model in ("unidepth_v2", "unik3d"):
        f_e = viz.focal_from_fov(55.0, ew)
        pts = viz.points_from_depth(depth_e, f_e)
        outs = [pts.transpose(2, 0, 1).astype(np.float32)[None]]
        for n in names[1:]:
            if n == "confidence":
                outs.append(np.clip(depth_e / depth_e.max(), 0, 1).astype(np.float32))
            elif n == "intrinsics":
                outs.append(np.float32([[f_e, 0, ew / 2], [0, f_e, eh / 2], [0, 0, 1]]))
        return outs

    if model in ("moge_2", "metric_anything"):
        # MoGe's own view-plane parameterisation, so recover_focal_shift solves
        # back to the shift this builds in (zero) instead of a made-up one.
        uv = pointmap.normalized_view_plane_uv(ew, eh).astype(np.float64)
        pts = np.concatenate([uv * depth_e[..., None], depth_e[..., None]], axis=-1)
        mask = np.ones((eh, ew), np.float32)
        mask[:2, :] = mask[-2:, :] = mask[:, :2] = mask[:, -2:] = 0.0
        outs = [pts.astype(np.float32)]
        for n in names[1:]:
            if n == "normal":
                nrm = np.zeros((eh, ew, 3), np.float32)
                nrm[..., 2] = 1.0
                outs.append(nrm)
            elif n == "mask":
                outs.append(mask)
            elif n == "metric_scale":
                outs.append(np.float32([1.0]))
        return outs

    # Everything left is a single map on the engine grid.
    if (spec.get("depth_scale") or "") == "relative":
        # A relative model claims inverse depth up to an affine transform, so a
        # synthetic one has to be given an arbitrary one.
        out = (1.0 / depth_e) * (2.0 + scale) + shift
    else:
        out = depth_e
    outs = [out.astype(np.float32)]
    for n in names[1:]:
        outs.append(np.zeros((eh, ew), np.float32))     # e.g. depth_anything_v3 sky
    return outs


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def distribution(depth):
    """The three numbers printed under a panel, or None if nothing is finite."""
    a = np.asarray(depth, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if not finite.size:
        return None
    return {"p02": float(np.percentile(finite, 2)),
            "median": float(np.median(finite)),
            "p98": float(np.percentile(finite, 98)),
            "valid_pct": float(finite.size) / a.size * 100.0}


class LiveRunner:
    """Engines, through the same common.py the benchmarks and evaluation use.

    NOT VERIFIED: there is no GPU and no engine directory on the machine this
    was written on, so this class has never been executed. Everything below it
    -- adapters, geometry, ranges, layout, point clouds -- runs identically
    from --from-saved and is checked by tests/test_viz.py.
    """

    def __init__(self):
        from core import common                                  # needs tensorrt
        import tensorrt as trt
        self.common = common
        self.trt = trt
        self._cache = {}

    def _engine(self, path):
        if path not in self._cache:
            with open(path, "rb") as f, \
                    self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR)) as rt:
                engine = rt.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"could not deserialize {path}")
            context = engine.create_execution_context()
            bufs = self.common.allocate_buffers(engine)
            self._cache[path] = (engine, context, bufs)
        return self._cache[path]

    def run(self, engine_path, blob):
        engine, context, (inputs, outputs, bindings, stream) = self._engine(engine_path)
        inputs[0].host = np.ascontiguousarray(blob)
        return self.common.do_inference(context, engine=engine, bindings=bindings,
                                        inputs=inputs, outputs=outputs, stream=stream)

    def close(self):
        for _engine, _context, (inputs, outputs, _bindings, stream) in \
                self._cache.values():
            self.common.free_buffers(inputs, outputs, stream)
        self._cache.clear()


# ---------------------------------------------------------------------------
# One (image, model) result
# ---------------------------------------------------------------------------

def _metric3d_canonical_for_display(raw, spec, eh, ew, oh, ow):
    """Restore Metric3D canonical depth to the source grid without inventing fx.

    Canonical depth is proportional to metric depth for one image, so it is
    valid for the qualitative shape gallery after per-model normalization. It
    is not metres and must never enter the shared-axis metric figure.
    """
    depth = np.asarray(raw[0], dtype=np.float64).reshape(eh, ew)
    geom = viz.input_geometry(spec, oh, ow, eh, ew)
    y0, y1, x0, x1 = geom.crop_box()
    depth = depth[y0:y1, x0:x1]
    depth = cv2.resize(depth, (ow, oh), interpolation=cv2.INTER_LINEAR)
    return np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)


def _depth_pro_focal_from_result(result, image_width):
    """Convert Depth Pro's predicted horizontal field of view to focal pixels."""
    raw = result.get("raw") or []
    if len(raw) < 2:
        raise ValueError("Depth Pro did not return fov_deg")
    fov_deg = float(np.asarray(raw[1]).reshape(-1)[0])
    if not np.isfinite(fov_deg) or not 0.0 < fov_deg < 180.0:
        raise ValueError(f"Depth Pro returned invalid fov_deg={fov_deg!r}")
    return viz.focal_from_fov(fov_deg, image_width), fov_deg


def evaluate_one(model, spec, card, adapt, image_bgr, source, runner=None,
                 saved_root=None, image_key="", fx=None):
    """Raw outputs and depth for one model on one image, or a stated failure."""
    eh, ew = card["input_h"], card["input_w"]
    oh, ow = image_bgr.shape[:2]
    if not eh or not ew:
        return {"model": model, "error": "no input size in spec or benchmark record"}

    res = {"model": model, "engine_h": eh, "engine_w": ew,
           "source_h": oh, "source_w": ow, "origin": source}

    try:
        if source == "saved":
            path = viz.find_saved(saved_root, image_key, model)
            if not path:
                return dict(res, error=f"no saved output under "
                                       f"{rel(saved_root)}")
            raw = viz.load_outputs(path, spec)
            res["saved_from"] = rel(path)
        elif source == "synthetic":
            raw = synthesize(model, spec, eh, ew, oh, ow, fx)
        else:
            path = card["engine_path"]
            if not path or not os.path.isfile(path):
                return dict(res, error="engine not on this machine: " + (path or "-"))
            blob = adapt[model]["pre"](image_bgr, eh, ew)
            raw = runner.run(path, blob)
            res["engine_path"] = path
    except Exception as e:
        return dict(res, error=f"{type(e).__name__}: {e}")

    # Copied, not referenced. do_inference hands back the pinned host buffers
    # themselves, so the next inference on the same engine would rewrite an
    # array this run had already put in a figure.
    res["raw"] = [np.array(a, copy=True) for a in raw]
    ctx = {"source": {"intrinsics": {"fx": fx}}} if fx else {}
    try:
        depth = adapt[model]["depth"](res["raw"], eh, ew, oh, ow, ctx)
        res["depth"] = np.asarray(depth, dtype=np.float64)
    except Exception as e:
        res["error"] = f"postprocess: {type(e).__name__}: {e}"
        if model == "metric3d_v2":
            try:
                res["display_depth"] = _metric3d_canonical_for_display(
                    res["raw"], spec, eh, ew, oh, ow)
                res["display_depth_kind"] = "canonical depth (shape only; not metres)"
            except Exception as display_error:
                res["display_error"] = (
                    f"{type(display_error).__name__}: {display_error}")
        return res

    res["stats"] = distribution(res["depth"])
    return res


def cloud_for(model, spec, res, image_rgb):
    """(points, colours, how it was made) or (None, None, why not).

    A point cloud needs a camera. Point-map models carry one implicitly, and
    depth_pro and unidepth_v2 return one. A model that returns only depth does
    not, and inventing a field of view for it would put a made-up room on
    screen next to measured ones.
    """
    eh, ew = res["engine_h"], res["engine_w"]
    raw = res.get("raw") or []
    layout = viz.POINT_LAYOUT.get(model)
    names = [o["name"] for o in spec["outputs"]]

    if layout and raw:
        pts = viz.points_from_point_map(raw[0], layout, eh, ew)
        if model in ("moge_2", "metric_anything"):
            mask_i = names.index("mask") if "mask" in names else None
            scale_i = names.index("metric_scale") if "metric_scale" in names else None
            mask = (np.asarray(raw[mask_i]).reshape(eh, ew) > 0.5
                    if mask_i is not None and mask_i < len(raw) else None)
            _, shift = pointmap.recover_focal_shift(pts, mask)
            s = (float(np.asarray(raw[scale_i]).reshape(-1)[0])
                 if scale_i is not None and scale_i < len(raw) else 1.0)
            pts = pts.copy()
            pts[..., 2] += shift
            pts *= s
            how = f"point map, shift {shift:+.4f}, metric_scale {s:.4g}"
        else:
            how = "point map, camera frame, as returned"
        col = cv2.resize(image_rgb, (ew, eh), interpolation=cv2.INTER_AREA)
        return pts, col, how

    if "fov_deg" in names and len(raw) > names.index("fov_deg"):
        fov = float(np.asarray(raw[names.index("fov_deg")]).reshape(-1)[0])
        d = res["depth"]
        f = viz.focal_from_fov(fov, d.shape[1])
        return viz.points_from_depth(d, f), image_rgb, \
            f"depth unprojected with the predicted {fov:.1f} deg fov"

    return None, None, "no camera in this model's outputs; nothing to unproject"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _skip_tile(h, w, text):
    """A tile that says why there is no picture, at the shape of the missing one.

    Rule 9: a failure is a result. It gets a panel the same size as everyone
    else's so it is counted when the figure is read, not quietly absent.
    """
    scale = max(420.0 / max(w, 1), 1.0)
    h, w = int(h * scale), int(w * scale)
    img = np.full((h, w, 3), 32, np.uint8)
    fs = 0.42
    step = 20
    y = 30
    for line in _wrap(text, max(18, int((w - 20) / (fs * 19)))):
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (215, 180, 180), 1, cv2.LINE_AA)
        y += step
    return img


def _wrap(text, width):
    words, lines, cur = str(text).split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        lines.append(cur)
    return lines[:6]


def depth_figure(inp, results, cards, skipped, synthetic, depth_range=None,
                 align_to=None, cols=4):
    """The comparison figure, and the JSON of every number printed on it."""
    depths = {r["model"]: r["depth"] for r in results if "depth" in r}
    fitted = {}
    if align_to and align_to in depths:
        ref_inv = 1.0 / np.clip(depths[align_to], 1e-6, None)
        for m, d in list(depths.items()):
            if cards[m]["band"] == "metric":
                continue
            try:
                s, t = viz.fit_scale_shift(d, ref_inv)
            except ValueError:
                continue
            aligned = s * d + t
            depths[m] = 1.0 / np.clip(aligned, 1e-6, None)
            # A fitted model gets the metric axis so it can be compared, and
            # its own section so it is not read as one. Its metres are borrowed
            # from the reference; it never claimed any.
            cards[m] = dict(cards[m], band="fitted", unit="m fitted")
            fitted[m] = {"scale": s, "shift": t, "reference": align_to}

    ranges, shared_note = viz.display_ranges(cards, depths, depth_range)
    shared = next((v for m, v in ranges.items() if cards[m]["band"] == "metric"), None)

    rgb = cv2.cvtColor(cv2.imread(inp["path"]), cv2.COLOR_BGR2RGB)
    sections = [viz.Section(
        header="input",
        panels=[viz.Panel(
            title=os.path.basename(inp["path"]),
            image=rgb, band="input",
            caption=(f"aspect {inp.get('aspect', '?')}  {rgb.shape[1]}x{rgb.shape[0]}\n"
                     + "\n".join(_wrap(inp.get("source", "") or "no source recorded", 74)[:2])),
        )])]

    per_band = {"metric": [], "fitted": [], "relative": [], "unknown": []}
    printed = {}
    for r in results:
        m = r["model"]
        card = cards[m]
        band = card["band"]
        if "depth" not in r:
            per_band.setdefault(band, []).append(viz.Panel(
                title=m, band="skipped",
                image=_skip_tile(rgb.shape[0], rgb.shape[1],
                                 r.get("error", "no result")),
                caption=viz.caption_for(card, [f"not drawn: {r.get('error', '')[:70]}"]),
            ))
            printed[m] = {"error": r.get("error")}
            continue
        vmin, vmax, cmap, how = ranges[m]
        # What is drawn is what was measured for the caption: `depths[m]`, not
        # the raw result, because a fitted model's array is not its raw one.
        shown = depths[m]
        stats = distribution(shown)
        extra = [how]
        if m in fitted:
            extra.append(f"scale+shift fitted to {align_to}: "
                         f"s={fitted[m]['scale']:.4g} t={fitted[m]['shift']:.4g}")
        elif band != "metric":
            extra.append("NOT aligned to anything; shape only, not distance")
        if stats:
            extra.append(f"p02 {stats['p02']:.3g}  med {stats['median']:.3g}  "
                         f"p98 {stats['p98']:.3g}  valid {stats['valid_pct']:.1f}%")
        if r.get("focal_source"):
            extra.append(f"fx {r['focal_length_px']:.2f} px from "
                         f"{r['focal_source']} (estimated intrinsics)")
        if card["mean_ms"]:
            extra.append(f"{card['mean_ms']:.2f} ms on {card.get('device') or '?'}")
        per_band[band].append(viz.Panel(
            title=m, band=band,
            image=viz.colorize(shown, vmin, vmax, cmap),
            caption=viz.caption_for(card, extra),
            badge=SYNTHETIC_BADGE if synthetic else "",
        ))
        printed[m] = {"vmin": vmin, "vmax": vmax, "cmap": cmap, "how": how,
                      "band": band, "stats": stats,
                      "engine_fingerprint": card["engine_fingerprint"],
                      "engine_fingerprint_source": card["engine_fingerprint_source"],
                      "input_size": [card["input_h"], card["input_w"]],
                       "precision": card["precision"], "unit": card["unit"],
                       "outputs": card["outputs"], "fit": fitted.get(m),
                       "focal_length_px": r.get("focal_length_px"),
                       "focal_source": r.get("focal_source"),
                       "fov_deg": r.get("fov_deg")}

    if per_band["metric"]:
        sections.append(viz.Section(
            header="metric depth -- one range, one unit, comparable",
            panels=per_band["metric"],
            note=shared_note if shared else "",
            colorbar={"vmin": shared[0], "vmax": shared[1], "cmap": shared[2],
                      "label": "depth (metres), same axis for every panel above"}
            if shared else None))
    if per_band["fitted"]:
        sections.append(viz.Section(
            header=f"relative depth, scale+shift fitted to {align_to} -- drawn "
                   f"on the metric axis, but the metres are borrowed",
            panels=per_band["fitted"],
            note="a relative model claims a shape, not a distance"))
    if per_band["relative"]:
        sections.append(viz.Section(
            header="relative depth -- own axis, NOT metres, not comparable to the above",
            panels=per_band["relative"],
            note="each panel min-max on itself"))
    if per_band["unknown"]:
        sections.append(viz.Section(
            header="global scale unknown -- the model does not claim one",
            panels=per_band["unknown"], note="own axis"))

    foot = [f"models not drawn: {len(skipped)}"] if skipped else []
    foot += [f"  {m}: {why}" for m, why in sorted(skipped.items())]
    if synthetic:
        foot.insert(0, "SYNTHETIC RUN: arrays are analytic, not model output. "
                       "Layout and scaling are real; the depth is not.")
    drawn = {b: sum(1 for p in ps if p.band == b) for b, ps in per_band.items()}
    failed = sum(len(ps) for ps in per_band.values()) - sum(drawn.values())
    sub = (f"{drawn['metric']} metric on a shared range | "
           f"{drawn['fitted']} relative fitted to {align_to or '-'} | "
           f"{drawn['relative']} relative on their own | "
           f"{drawn['unknown']} scale unknown | "
           f"{failed} ran but produced no depth | {len(skipped)} excluded "
           f"before running")
    fig = viz.render_figure(sections, f"depth comparison -- {inp['key']}",
                            subtitle=sub, footer="\n".join(foot), cols=cols)
    return fig, {"shared_metric_range": shared[:2] if shared else None,
                 "shared_range_note": shared_note if shared else None,
                 "per_model": printed, "not_drawn": skipped}


def qualitative_depth_figure(inp, results, cards, skipped, synthetic, cols=5):
    """Compact shape comparison for the project README.

    Unlike :func:`depth_figure`, this deliberately does not compare metres.
    Every valid model is normalized on its own robust range and drawn with one
    colour map and one polarity (warm=near). The metric-audit figure remains
    available through ``--view metric``.
    """
    rgb = cv2.cvtColor(cv2.imread(inp["path"]), cv2.COLOR_BGR2RGB)
    panels = [viz.Panel(
        title="Input",
        image=rgb,
        band="input",
        caption=f"{rgb.shape[1]}x{rgb.shape[0]}",
    )]
    printed = {}
    for r in results:
        model = r["model"]
        card = cards[model]
        shown_depth = r.get("depth")
        if shown_depth is None:
            shown_depth = r.get("display_depth")
        if shown_depth is None:
            error = r.get("error", "no result")
            panels.append(viz.Panel(
                title=model,
                image=_skip_tile(rgb.shape[0], rgb.shape[1], error),
                band="skipped",
                caption="not available",
            ))
            printed[model] = {"error": error}
            continue

        shown, vmin, vmax = viz.qualitative_display(
            shown_depth, card.get("output_form", "depth"))
        timing = (f" · {card['mean_ms']:.2f} ms" if card.get("mean_ms") else "")
        if r.get("display_depth_kind"):
            scale_label = "canonical"
        elif r.get("focal_source") == "depth_pro predicted fov_deg":
            scale_label = "metric (Depth Pro fx)"
        else:
            scale_label = card["depth_scale"]
        panels.append(viz.Panel(
            title=model,
            image=viz.colorize(shown, vmin, vmax, viz.QUALITATIVE_CMAP),
            band="qualitative",
            caption=f"{scale_label} · own 2-98% range{timing}",
            badge=SYNTHETIC_BADGE if synthetic else "",
        ))
        printed[model] = {
            "display": "proximity",
            "polarity": "warm=near, cool=far",
            "normalization": "per-model 2-98 percentile",
            "source_output_form": card.get("output_form", "depth"),
            "vmin": vmin,
            "vmax": vmax,
            "cmap": viz.QUALITATIVE_CMAP,
            "stats": distribution(shown_depth),
            "engine_fingerprint": card["engine_fingerprint"],
            "engine_fingerprint_source": card["engine_fingerprint_source"],
            "input_size": [card["input_h"], card["input_w"]],
            "precision": card["precision"],
            "focal_length_px": r.get("focal_length_px"),
            "focal_source": r.get("focal_source"),
            "fov_deg": r.get("fov_deg"),
        }
        if r.get("display_depth_kind"):
            printed[model]["display_source"] = r["display_depth_kind"]
            printed[model]["metric_conversion_error"] = r.get("error")

    foot = [
        "Qualitative view: every model uses its own robust range; colours do not compare metres.",
        "Polarity is unified: warm=near, cool=far. Use --view metric for the shared metric axis.",
    ]
    if skipped:
        foot.append("Not drawn: " + "; ".join(
            f"{m} ({why})" for m, why in sorted(skipped.items())))
    if synthetic:
        foot.insert(0, "SYNTHETIC RUN: analytic arrays, not model output.")
    section = viz.Section(
        header="input and model outputs -- shape comparison only",
        panels=panels,
        note="same Turbo map · warm=near · per-model 2-98%",
    )
    fig = viz.render_figure(
        [section],
        f"monocular depth gallery -- {inp['key']}",
        subtitle=(f"{len(printed)} model attempts · compact qualitative view; "
                  "absolute colour values are intentionally not shared"),
        footer="\n".join(foot),
        cols=cols,
        panel_in=2.35,
    )
    return fig, {
        "view": "qualitative",
        "shared_metric_range": None,
        "per_model": printed,
        "not_drawn": skipped,
    }


def inputs_figure(inp, results, cards, specs, adapt, cols=4):
    """What each model actually gets fed, with the pad region drawn on it."""
    bgr = cv2.imread(inp["path"])
    oh, ow = bgr.shape[:2]
    panels, recorded = [], {}
    for r in results:
        m = r["model"]
        eh, ew = r.get("engine_h"), r.get("engine_w")
        if not eh:
            continue
        try:
            geom = viz.input_geometry(specs[m], oh, ow, eh, ew)
        except ValueError as e:
            geom = None
            note = str(e)
        else:
            note = viz.geometry_note(geom)
        try:
            blob = adapt[m]["pre"](bgr, eh, ew)
            img = viz.preview_input_tensor(blob)
        except Exception as e:
            img = _skip_tile(eh or 200, ew or 260,
                             f"preprocess unavailable here: "
                             f"{type(e).__name__}: {e}")
            geom = None
        panels.append(viz.Panel(
            title=m, band="input", image=img,
            caption=viz.caption_for(cards[m], [note]),
            box=geom.crop_box() if geom else None,
            box_label="image" if geom and any(
                (geom.pad_top, geom.pad_bottom, geom.pad_left, geom.pad_right)) else "",
        ))
        recorded[m] = {"engine_size": [eh, ew], "geometry": note}
    fig = viz.render_figure(
        [viz.Section(header="network input as fed, dashed box = the image inside "
                            "the padding", panels=panels,
                     note=f"source {ow}x{oh}")],
        f"preprocessing -- {inp['key']}",
        subtitle="per-tensor min-max here is fine: this is the input, not a measurement",
        cols=cols)
    return fig, recorded


def outputs_figure(inp, results, cards, specs, synthetic, cols=4):
    """Every declared output that is not the depth map: mask, normal, camera."""
    oh, ow = cv2.imread(inp["path"]).shape[:2]
    panels, recorded = [], {}
    for r in results:
        m = r["model"]
        raw = r.get("raw")
        if not raw:
            continue
        eh, ew = r["engine_h"], r["engine_w"]
        entries = []
        for i, (name, meaning) in enumerate(cards[m]["outputs"][1:], start=1):
            if i >= len(raw):
                break
            try:
                img, text = viz.view_of_output(name, meaning, raw[i], eh, ew,
                                               min(oh, 480), int(min(oh, 480) * ow / oh))
            except Exception as e:
                img, text = None, f"{meaning}: {type(e).__name__}: {e}"
            entries.append((name, text))
            if img is not None:
                panels.append(viz.Panel(
                    title=f"{m} / {name}", band="input", image=img,
                    caption=viz.caption_for(cards[m], [text]),
                    badge=SYNTHETIC_BADGE if synthetic else ""))
        if entries:
            recorded[m] = dict(entries)
    if not panels and not recorded:
        return None, {}
    scalars = [f"{m}: " + "; ".join(f"{k} -> {v}" for k, v in d.items())
               for m, d in sorted(recorded.items())]
    fig = viz.render_figure(
        [viz.Section(header="other declared outputs (spec.json decides what exists)",
                     panels=panels)],
        f"auxiliary outputs -- {inp['key']}",
        subtitle="only outputs a model actually declares are shown",
        footer="\n".join(scalars[:14]), cols=cols)
    return fig, recorded


def cloud_figure(inp, results, cards, specs, out_dir, synthetic, cols=4,
                 write_ply=True):
    rgb = cv2.cvtColor(cv2.imread(inp["path"]), cv2.COLOR_BGR2RGB)
    panels, recorded = [], {}
    for r in results:
        m = r["model"]
        if "depth" not in r:
            continue
        try:
            pts, col, how = cloud_for(m, specs[m], r, rgb)
        except Exception as e:
            pts, col, how = None, None, f"{type(e).__name__}: {e}"
        if pts is None:
            recorded[m] = {"cloud": None, "why": how}
            continue
        img = viz.render_points(pts, col)
        ply = ""
        if write_ply:
            ply = os.path.join(out_dir, f"{inp['key']}_{m}.ply")
            viz.write_ply(ply, pts, col)
            ply = rel(ply)
        n = int((np.isfinite(pts).all(-1) & (pts[..., 2] > 0)).sum())
        recorded[m] = {"cloud": ply or None, "points": n, "how": how}
        panels.append(viz.Panel(
            title=f"{m} point cloud", band=cards[m]["band"], image=img,
            caption=viz.caption_for(cards[m], [how, f"{n} points" +
                                               (f" -> {ply}" if ply else "")]),
            badge=SYNTHETIC_BADGE if synthetic else ""))
    if not panels:
        return None, recorded
    foot = [f"  {m}: {d['why']}" for m, d in sorted(recorded.items()) if not d["cloud"]]
    fig = viz.render_figure(
        [viz.Section(header="point clouds -- only models that return a camera "
                            "or a point map", panels=panels)],
        f"point clouds -- {inp['key']}",
        subtitle="orthographic, z-buffered, viewed from above-left",
        footer=("no cloud for:\n" + "\n".join(foot)) if foot else "", cols=cols)
    return fig, recorded


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    specs = spec_mod.load_all()
    if not specs:
        raise SystemExit("no specs under models/")
    adapt, why = adapters()
    if not adapt:
        raise SystemExit(f"could not read evaluate_gt.ADAPTERS ({why})")

    wanted = [m.strip() for m in args.models.split(",")] if args.models else None
    models, skipped = select_models(specs, adapt, wanted)
    if not models:
        raise SystemExit("no model left to draw")
    if args.metric3d_fx_from_depth_pro:
        missing = [m for m in ("depth_pro", "metric3d_v2") if m not in models]
        if missing:
            raise SystemExit("--metric3d-fx-from-depth-pro requires both "
                             "depth_pro and metric3d_v2 in the selected models; "
                             "missing: " + ", ".join(missing))

    runs = bench_runs()
    cards = {m: viz.model_card(m, specs[m], runs.get(m)) for m in models}
    for m in list(models):
        if not cards[m]["input_h"]:
            skipped[m] = "no input size in spec or benchmark record"
            models.remove(m)

    inputs, where = resolve_inputs(args.image)
    print(f"inputs: {where}")
    print(f"models: {len(models)} drawn, {len(skipped)} not")
    for m, w in sorted(skipped.items()):
        print(f"  skip {m}: {w}")

    source = ("saved" if args.from_saved else
              "synthetic" if args.synthetic else
              "live" if args.live else "")
    if not source:
        raise SystemExit(
            "pick a source of model output: --from-saved DIR (arrays saved "
            "earlier), --live (engines on this machine), or --synthetic "
            "(analytic arrays, for checking the layout without a GPU).")
    print(f"source: {source}")

    runner = None
    if source == "live":
        try:
            runner = LiveRunner()
        except Exception as e:
            raise SystemExit(f"--live needs TensorRT and common.py here: "
                             f"{type(e).__name__}: {e}")

    view = "metric" if (args.depth_range or args.align_relative_to) else args.view
    cols = args.cols or (5 if view == "qualitative" else 4)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    saved_root = args.from_saved or os.path.join(out_dir, "saved")
    written, summary = [], {"inputs_from": where, "source": source,
                            "depth_view": view,
                            "models_not_drawn": skipped, "images": {}}

    try:
        for inp in inputs:
            bgr = cv2.imread(inp["path"])
            if bgr is None:
                print(f"  cannot read {inp['path']}")
                continue
            results = []
            depth_pro_focal = None
            depth_pro_focal_error = None
            for m in models:
                effective_fx = args.fx
                if m == "metric3d_v2" and args.metric3d_fx_from_depth_pro:
                    effective_fx = (depth_pro_focal or {}).get("fx")
                r = evaluate_one(m, specs[m], cards[m], adapt, bgr, source,
                                 runner=runner, saved_root=saved_root,
                                 image_key=inp["key"], fx=effective_fx)
                if m == "depth_pro" and args.metric3d_fx_from_depth_pro:
                    try:
                        fx_px, fov_deg = _depth_pro_focal_from_result(
                            r, bgr.shape[1])
                        depth_pro_focal = {"fx": fx_px, "fov_deg": fov_deg}
                        print(f"  {inp['key']:<18} {'depth_pro focal':<20} "
                              f"{fx_px:.2f} px from {fov_deg:.3f} deg")
                    except Exception as e:
                        depth_pro_focal_error = f"{type(e).__name__}: {e}"
                if m == "metric3d_v2" and args.metric3d_fx_from_depth_pro:
                    if depth_pro_focal:
                        r["focal_length_px"] = depth_pro_focal["fx"]
                        r["fov_deg"] = depth_pro_focal["fov_deg"]
                        r["focal_source"] = "depth_pro predicted fov_deg"
                    else:
                        r["error"] = ("Depth Pro focal estimate unavailable: "
                                      + (depth_pro_focal_error or "no result"))
                results.append(r)
                mark = "ok" if "depth" in r else "-- " + str(r.get("error", ""))[:60]
                print(f"  {inp['key']:<18} {m:<20} {mark}")
                if args.save_outputs and r.get("raw") is not None:
                    viz.save_outputs(
                        os.path.join(args.save_outputs, inp["key"], m + ".npz"),
                        m, specs[m], r["raw"],
                        {"image": rel(inp["path"]),
                         "engine_size": [r["engine_h"], r["engine_w"]],
                         "source": source})

            per_image = {"path": rel(inp["path"]),
                          "aspect": inp.get("aspect"), "figures": {}}
            if args.metric3d_fx_from_depth_pro:
                per_image["metric3d_focal"] = (
                    {"focal_length_px": depth_pro_focal["fx"],
                     "fov_deg": depth_pro_focal["fov_deg"],
                     "source": "depth_pro predicted fov_deg",
                     "status": "estimated intrinsics, not measured calibration"}
                    if depth_pro_focal else {"error": depth_pro_focal_error})
            if view == "qualitative":
                fig, printed = qualitative_depth_figure(
                    inp, results, dict(cards), skipped,
                    source == "synthetic", cols)
            else:
                fig, printed = depth_figure(
                    inp, results, dict(cards), skipped,
                    source == "synthetic", args.depth_range,
                    args.align_relative_to, cols)
            p = viz.save_figure(fig, os.path.join(out_dir, f"{inp['key']}_depth.png"))
            written.append(p)
            per_image["figures"]["depth"] = rel(p)
            per_image["depth"] = printed

            if args.inputs_figure:
                fig, rec = inputs_figure(inp, results, cards, specs, adapt, cols)
                p = viz.save_figure(fig, os.path.join(out_dir,
                                                      f"{inp['key']}_inputs.png"))
                written.append(p)
                per_image["figures"]["inputs"] = rel(p)
                per_image["preprocess"] = rec

            if args.extras:
                fig, rec = outputs_figure(inp, results, cards, specs,
                                          source == "synthetic", cols)
                if fig is not None:
                    p = viz.save_figure(fig, os.path.join(
                        out_dir, f"{inp['key']}_outputs.png"))
                    written.append(p)
                    per_image["figures"]["outputs"] = rel(p)
                per_image["auxiliary_outputs"] = rec

            if args.pointcloud:
                fig, rec = cloud_figure(inp, results, cards, specs, out_dir,
                                        source == "synthetic", cols,
                                        write_ply=not args.no_ply)
                if fig is not None:
                    p = viz.save_figure(fig, os.path.join(
                        out_dir, f"{inp['key']}_cloud.png"))
                    written.append(p)
                    per_image["figures"]["cloud"] = rel(p)
                per_image["point_clouds"] = rec

            summary["images"][inp["key"]] = per_image
    finally:
        if runner is not None:
            runner.close()

    spath = os.path.join(out_dir, "demo_summary.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    written.append(spath)

    print(f"\n{len(written)} files under {rel(out_dir)}")
    for p in written:
        print("  " + rel(p))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("where the model output comes from")
    src.add_argument("--from-saved", metavar="DIR",
                     help="read arrays saved by an earlier run (no GPU needed)")
    src.add_argument("--live", action="store_true",
                     help="run the engines on this machine (needs TensorRT)")
    src.add_argument("--synthetic", action="store_true",
                     help="analytic arrays; checks the drawing, not the models")
    src.add_argument("--save-outputs", metavar="DIR",
                     help="write every raw output as npz for a later --from-saved")

    ap.add_argument("--image", action="append",
                    help="an input image; repeat for several. Default is "
                         "data/eval/aspects.json, else data/example.jpg")
    ap.add_argument("--models", help="comma separated subset")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--view", choices=("qualitative", "metric"),
                    default="qualitative",
                    help="depth figure style: compact per-model shape gallery "
                         "(default) or shared-axis metric audit")
    ap.add_argument("--cols", type=int,
                    help="panels per row (default: 5 qualitative, 4 metric)")
    ap.add_argument("--depth-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                    help="fix the shared metric range in metres")
    ap.add_argument("--align-relative-to", metavar="MODEL",
                    help="fit each relative model's scale and shift to this "
                         "model and draw it on the metric axis, saying so")
    focal = ap.add_mutually_exclusive_group()
    focal.add_argument("--fx", type=float,
                       help="measured focal length in pixels of the camera that "
                            "took the image")
    focal.add_argument("--metric3d-fx-from-depth-pro", action="store_true",
                       help="derive Metric3D fx from Depth Pro's predicted FOV; "
                            "this is estimated intrinsics, not calibration")
    ap.add_argument("--no-inputs-figure", dest="inputs_figure",
                    action="store_false", help="skip the preprocessing figure")
    ap.add_argument("--no-extras", dest="extras", action="store_false",
                    help="skip the auxiliary output figure")
    ap.add_argument("--no-pointcloud", dest="pointcloud", action="store_false",
                    help="skip point clouds")
    ap.add_argument("--no-ply", action="store_true",
                    help="render clouds but do not write .ply files")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
