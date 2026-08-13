"""Score engines against measured depth.

    python evaluate_gt.py --manifest data/eval/diode_indoors.json
    python evaluate_gt.py --manifest ... --check      # adapters only, no GPU
    python evaluate_gt.py --manifest ... depth_pro

The comparison table says how fast each engine is and how closely it matches
the graph it was built from. Neither says whether the depth is right. This
does, against DIODE: real measurements in metres, with a mask saying which
pixels the scanner actually reached.

**The adapters are the risky part, so they are checked against an oracle.**
Getting a model onto ground truth means reproducing its preprocessing exactly
-- channel order, the divide by 255, whether ImageNet statistics apply, which
interpolation, and what the network's output even means -- and every one of
those is a silent failure. Feed a BGR image to a model trained on RGB and it
still returns a plausible depth map that scores badly, and the conclusion
lands on the model.

So each adapter is first run on data/example.jpg and compared against
reports/inputs/<model>.npy, the byte-exact tensor that model's own onnx2trt.py
fed the engine when it was benchmarked. An adapter that cannot reproduce that
does not get to score anything. `--check` runs only that part and needs no GPU.

**Alignment is decided by the model's contract, never by the numbers.** A
model that claims metres is scored in metres. See core/gt.py.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core import bench, gt, spec as spec_mod  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(ROOT, "reports", "inputs")
OUT_DIR = os.path.join(ROOT, "reports", "gt")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Adapters: one per model, transcribed from its own onnx2trt.py.
#
# `pre` takes the original BGR image and returns exactly what that model feeds
# its engine. `depth` takes the engine's raw outputs and the original size and
# returns depth in metres on the original pixel grid.
# --------------------------------------------------------------------------

def _imagenet_square(bgr, size):
    """depth_anything_v2 and _v3: stretch to a square, then ImageNet."""
    img = cv2.resize(bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None]).astype(np.float32)


def _direct_depth(outs, shape, orig_h, orig_w):
    """The output is depth already; put it back on the original grid."""
    d = np.asarray(outs[0]).reshape(shape).astype(np.float64)
    d = cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(d, 1e-3, 1e3)


def _depth_pro_pre(bgr, size=1536):
    """Normalise first, resize second -- the order upstream uses.

    Both steps are linear, so the order does not change the values much, but
    it changes them, and the oracle is byte-exact. torch's bilinear with
    align_corners=False is not cv2's, either.
    """
    import torch

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
    x = ((x - 0.5) / 0.5).unsqueeze(0)
    if x.shape[-2:] != (size, size):
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                            align_corners=False)
    return np.ascontiguousarray(x.numpy())


def _depth_pro_depth(outs, shape, orig_h, orig_w):
    """canonical inverse depth * (W / f_px), then inverted.

    f_px comes from the field of view the model predicts, so nothing per-image
    has to be supplied -- and W is the *original* width, not the network's,
    which is what makes the metres metres.
    """
    canonical = np.asarray(outs[0]).reshape(shape).astype(np.float64)
    fov_deg = float(np.asarray(outs[1]).reshape(-1)[0])
    f_px = 0.5 * orig_w / np.tan(0.5 * np.deg2rad(fov_deg))
    inv = canonical * (orig_w / f_px)
    inv = cv2.resize(inv, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return 1.0 / np.clip(inv, 1e-4, 1e4)


def _rgb_over_255(bgr, h, w, interp=cv2.INTER_LINEAR):
    """BGR -> RGB, resize, /255. No ImageNet statistics."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (w, h), interpolation=interp).astype(np.float64) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None]).astype(np.float32)


def _zipdepth_pre(bgr, h=384, w=512):
    """/255 and nothing else. No ImageNet statistics -- see its spec.json."""
    img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


_RECOVER = None


def _recover_focal_shift():
    """MoGe's solver, from whichever clone is on this machine.

    moge_2 and metric_anything both end in the same place: the network returns
    a point map whose Z is only depth once a shift has been solved for, and
    that solver lives upstream. Reimplementing it here would be a second
    version of the thing being measured.
    """
    global _RECOVER
    if _RECOVER is not None:
        return _RECOVER
    candidates = [
        (os.path.join(ROOT, "models", "moge_2"), "MoGe.moge.utils.geometry_torch"),
        (os.path.join(ROOT, "models", "metric_anything", "metric_anything",
                      "models", "student_pointmap"), "moge.utils.geometry_torch"),
    ]
    errors = []
    for path, module in candidates:
        if not os.path.isdir(path):
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            mod = __import__(module, fromlist=["recover_focal_shift"])
            _RECOVER = mod.recover_focal_shift
            return _RECOVER
        except Exception as e:                                # noqa: BLE001
            errors.append(f"{module}: {type(e).__name__}: {e}")
    raise ImportError("recover_focal_shift not importable; tried " + "; ".join(errors))


def _point_map_depth(outs, shape, orig_h, orig_w, mask_idx, scale_idx):
    """Point-map Z, shifted and scaled into metres, on the original grid.

    Transcribed from the two models' own post-processing, which is identical:
    solve for the shift, add it to Z, multiply by the predicted metric scale.
    The mask the model returns is deliberately *not* applied -- the evaluation
    has its own mask from the scanner, and letting a model mark its own pixels
    invalid would let it drop the ones it finds hard.
    """
    import torch

    h, w = shape
    points = torch.from_numpy(np.asarray(outs[0]).reshape(1, h, w, 3).copy())
    mask = torch.from_numpy(np.asarray(outs[mask_idx]).reshape(1, h, w).copy()) > 0.5
    metric_scale = torch.from_numpy(np.asarray(outs[scale_idx]).reshape(1).copy())

    _, shift = _recover_focal_shift()(points, mask)
    depth = (points[..., 2] + shift[..., None, None]) * metric_scale[:, None, None]
    d = depth[0].double().numpy()
    d = cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.where(d > 0, d, np.nan)


ADAPTERS = {
    "depth_anything_v2": {
        "pre": lambda bgr: _imagenet_square(bgr, 518),
        "depth": lambda outs, oh, ow: _direct_depth(outs, (518, 518), oh, ow),
    },
    "depth_anything_v3": {
        "pre": lambda bgr: _imagenet_square(bgr, 518),
        "depth": lambda outs, oh, ow: _direct_depth(outs, (518, 518), oh, ow),
    },
    "depth_pro": {
        "pre": _depth_pro_pre,
        "depth": lambda outs, oh, ow: _depth_pro_depth(outs, (1536, 1536), oh, ow),
    },
    "depth_anything_ac": {
        "pre": lambda bgr: _imagenet_square(bgr, 518),
        "depth": lambda outs, oh, ow: _direct_depth(outs, (518, 518), oh, ow),
    },
    "distill_any_depth": {
        "pre": lambda bgr: _imagenet_square(bgr, 518),
        "depth": lambda outs, oh, ow: _direct_depth(outs, (518, 518), oh, ow),
    },
    "zipdepth": {
        "pre": _zipdepth_pre,
        "depth": lambda outs, oh, ow: _direct_depth(outs, (384, 512), oh, ow),
    },
    # points, normal, mask, metric_scale
    "moge_2": {
        "pre": lambda bgr: _rgb_over_255(bgr, 388, 518),
        "depth": lambda outs, oh, ow: _point_map_depth(outs, (388, 518), oh, ow, 2, 3),
    },
    # points, mask, metric_scale -- no normal branch
    "metric_anything": {
        "pre": lambda bgr: _rgb_over_255(bgr, 388, 518, cv2.INTER_AREA),
        "depth": lambda outs, oh, ow: _point_map_depth(outs, (388, 518), oh, ow, 1, 2),
    },
}


def check_adapter(model, example_bgr):
    """Does this adapter reproduce the tensor the benchmark actually fed?"""
    oracle_path = os.path.join(INPUTS, f"{model}.npy")
    if not os.path.exists(oracle_path):
        return False, f"no {os.path.relpath(oracle_path, ROOT)} to check against"
    oracle = np.load(oracle_path)
    got = ADAPTERS[model]["pre"](example_bgr)
    if got.shape != oracle.shape:
        return False, f"shape {got.shape} against the recorded {oracle.shape}"
    diff = float(np.abs(got.astype(np.float64) - oracle.astype(np.float64)).max())
    # Not exact equality: the recorded tensor went through float32 arithmetic in
    # a different order. A mismatch from a wrong channel order or a missing
    # normalisation is orders of magnitude larger than this.
    return diff < 1e-4, f"max|diff| {diff:.3e}"


def load_engine(path):
    import tensorrt as trt
    with open(path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.ERROR)) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"could not deserialize {path}")
    return engine


def score_model(model, manifest, root, runs, limit=None, diagnose=False):
    import common

    run = runs[model]
    engine = load_engine(run["engine_path"])
    sp = spec_mod.load_all()[model]
    policy = gt.policy_for(sp.get("depth_scale"))
    if policy is None:
        return {"model": model, "skip": "no alignment policy for this output contract"}
    # Which way round a relative output runs decides whether the scale+shift fit
    # is affine in the space the model was trained in. There is no safe default,
    # so it is required rather than guessed, and checked against the data below.
    form = sp.get("output_form")
    # Undeclared is not scored -- but it is measured, so the run tells you what
    # to declare instead of leaving you to guess the thing it just refused to
    # guess for you. A few images settle a sign.
    orientation_only = policy == "scale_shift" and form not in ("depth", "inverse_depth")
    if orientation_only:
        limit = min(limit or 5, 5)
    pred_is_inverse = form == "inverse_depth"

    per_sample, fits = [], []
    with engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        samples = manifest["samples"][:limit] if limit else manifest["samples"]
        for s in samples:
            bgr = cv2.imread(os.path.join(root, s["rgb"]))
            if bgr is None:
                continue
            oh, ow = bgr.shape[:2]
            inputs[0].host = ADAPTERS[model]["pre"](bgr)
            raw = common.do_inference(context, engine=engine, bindings=bindings,
                                      inputs=inputs, outputs=outputs, stream=stream)
            pred = ADAPTERS[model]["depth"](raw, oh, ow)

            g = np.load(os.path.join(root, s["depth"]))
            m = np.load(os.path.join(root, s["mask"])).astype(bool)
            g = g[..., 0] if g.ndim == 3 else g
            m = m[..., 0] if m.ndim == 3 else m
            m = m & (g > 0)
            if orientation_only:
                per_sample.append({"id": s["id"], "n": 0,
                                   "orientation": gt.orientation(pred, g, m)})
                continue
            try:
                aligned, fit = gt.align(pred, g, m, policy, pred_is_inverse)
            except ValueError as e:
                per_sample.append({"id": s["id"], "skip": str(e)})
                continue
            r = gt.metrics(aligned, g, m, max_depth=manifest.get("max_depth"))
            r["id"] = s["id"]
            r["orientation"] = gt.orientation(pred, g, m)
            if diagnose:
                # What the score would be if the model were allowed a fit it
                # does not claim. This is never the published number -- it is
                # the question "is this a wrong scale or a wrong shape?", and
                # the two look identical in AbsRel alone. A metric model whose
                # error collapses under a scale fit is mis-scaled, not blind.
                for alt in ("scale", "scale_shift"):
                    if alt == policy:
                        continue
                    try:
                        a2, f2 = gt.align(pred, g, m, alt, pred_is_inverse)
                    except ValueError:
                        continue
                    r[f"diag_{alt}"] = gt.metrics(
                        a2, g, m, max_depth=manifest.get("max_depth"))
                    r[f"diag_{alt}_fit"] = f2
            per_sample.append(r)
            if fit:
                fits.append(fit)
        common.free_buffers(inputs, outputs, stream)

    if orientation_only:
        seen = [r["orientation"] for r in per_sample
                if r.get("orientation") == r.get("orientation")]
        if not seen:
            return {"model": model, "skip": "output_form undeclared and unmeasurable"}
        med = float(np.median(seen))
        measured = "depth" if med > 0 else "inverse_depth"
        return {"model": model, "orientation_median": med,
                "output_form_measured": measured,
                "skip": f"output_form undeclared. Measured over {len(seen)} images: "
                        f"correlation {med:+.3f} with true depth, so it is "
                        f"'{measured}'. Put that in models/{model}/spec.json."}

    scored = [r for r in per_sample if r.get("n")]
    if not scored:
        return {"model": model, "skip": "nothing scored"}
    # Averaged over images, not over pixels: an image with more valid pixels
    # should not count for more than another image.
    agg = {k: float(np.mean([r[k] for r in scored]))
           for k in ("abs_rel", "rmse", "log10", "delta1", "delta2", "delta3")}
    for alt in ("scale", "scale_shift"):
        got = [r[f"diag_{alt}"] for r in scored if f"diag_{alt}" in r]
        if got:
            agg[f"diag_{alt}"] = {k: float(np.mean([g2[k] for g2 in got]))
                                  for k in ("abs_rel", "delta1")}
            fitted = [r[f"diag_{alt}_fit"] for r in scored
                      if f"diag_{alt}_fit" in r]
            if fitted:
                agg[f"diag_{alt}"]["fit_median"] = {
                    k: float(np.median([f[k] for f in fitted])) for k in fitted[0]}
    orients = [r["orientation"] for r in scored
               if r.get("orientation") == r.get("orientation")]
    if orients:
        agg["orientation_median"] = float(np.median(orients))
        # Positive means the prediction rises with true depth, so it is a depth
        # map; negative means it falls, so it is a disparity map. A declaration
        # that disagrees with the data is reported rather than obeyed.
        measured = "depth" if agg["orientation_median"] > 0 else "inverse_depth"
        agg["output_form_declared"] = form
        agg["output_form_measured"] = measured
        if form and measured != form:
            agg["output_form_conflict"] = (
                f"spec.json says {form} but the prediction correlates "
                f"{agg['orientation_median']:+.3f} with true depth")
    agg.update(model=model, images=len(scored),
               pixels=int(sum(r["n"] for r in scored)),
               alignment=policy, engine_path=run["engine_path"],
               precision=run.get("precision"), samples=per_sample)
    if fits:
        agg["fit_median"] = {k: float(np.median([f[k] for f in fits]))
                             for k in fits[0]}
    return agg


def render(results, manifest):
    src = manifest["source"]
    L = [f"# Depth against ground truth", "",
         f"{src['dataset']} {manifest['split']}, {manifest['count']} images, "
         f"depth in {manifest['depth_units']} as {manifest['depth_convention']}. "
         f"Generated by `evaluate_gt.py`; the image list and its checksums are "
         f"in the manifest.", ""]
    ok = [r for r in results if not r.get("skip")]
    if ok:
        head = ["model", "alignment", "AbsRel", "RMSE", "log10",
                "d1", "d2", "d3", "images"]
        rows = [[f"`{r['model']}`", r["alignment"],
                 f"{r['abs_rel'] * 100:.2f}%", f"{r['rmse']:.3f}",
                 f"{r['log10']:.4f}", f"{r['delta1'] * 100:.1f}%",
                 f"{r['delta2'] * 100:.1f}%", f"{r['delta3'] * 100:.1f}%",
                 str(r["images"])]
                for r in sorted(ok, key=lambda r: r["abs_rel"])]
        w = [max(len(str(c)) for c in [h] + [r[i] for r in rows])
             for i, h in enumerate(head)]
        L.append("| " + " | ".join(h.ljust(x) for h, x in zip(head, w)) + " |")
        L.append("| " + " | ".join("---" if i < 2 else "---:"
                                   for i in range(len(head))) + " |")
        for r in rows:
            L.append("| " + " | ".join(str(c).ljust(x) for c, x in zip(r, w)) + " |")
        L += ["", "Best first by AbsRel. Averaged over images rather than over "
                  "pixels, so an image with more valid pixels does not count "
                  "for more than another.", ""]
    skipped = [r for r in results if r.get("skip")]
    if skipped:
        L += ["## Not scored", ""]
        L += [f"- `{r['model']}` -- {r['skip']}" for r in skipped]
        L.append("")
    L += ["**Alignment** is the model's own claim, not a choice made here. "
          "`none` means the output was compared in metres as it came out; "
          "`scale_shift` means a scale and a shift were fitted on disparity "
          "first, which is all a relative model claims. Two rows with "
          "different alignments are not comparable.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="default: every model with an adapter")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=None,
                    help="dataset root; default: the manifest's own directory tree")
    ap.add_argument("--limit", type=int, default=None, help="first N images only")
    ap.add_argument("--diagnose", action="store_true",
                    help="also report what the score would be under an alignment "
                         "the model does not claim, to tell a wrong scale from "
                         "a wrong shape. Never the published number.")
    ap.add_argument("--check", action="store_true",
                    help="verify the adapters against the recorded inputs and stop")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    wanted = args.models or sorted(ADAPTERS)
    unknown = [m for m in wanted if m not in ADAPTERS]
    if unknown:
        print(f"no adapter for {', '.join(unknown)} -- "
              f"have {', '.join(sorted(ADAPTERS))}")
        return 2

    example = cv2.imread(os.path.join(ROOT, "data", "example.jpg"))
    if example is None:
        print("data/example.jpg is missing; the adapters cannot be checked")
        return 1

    print("adapter check (against the tensor each benchmark actually fed):")
    good = []
    for m in wanted:
        passed, detail = check_adapter(m, example)
        print(f"  {'ok  ' if passed else 'FAIL'} {m:20} {detail}")
        if passed:
            good.append(m)
    if args.check:
        return 0 if len(good) == len(wanted) else 1
    if not good:
        print("no adapter reproduces its recorded input; nothing scored")
        return 1

    root = args.root or os.path.dirname(os.path.abspath(args.manifest))
    runs = {}
    for r in bench.load_all(os.path.join(ROOT, "reports", "bench")):
        runs[r["model"]] = r

    results = []
    for m in good:
        if m not in runs:
            results.append({"model": m, "skip": "not benchmarked"})
            continue
        if not os.path.exists(runs[m].get("engine_path", "")):
            results.append({"model": m, "skip": "engine not on this machine"})
            continue
        print(f"\n[{m}] scoring")
        r = score_model(m, manifest, root, runs, args.limit, args.diagnose)
        results.append(r)
        if r.get("skip"):
            print(f"  skipped: {r['skip']}")
        else:
            print(f"  AbsRel {r['abs_rel'] * 100:.2f}%  RMSE {r['rmse']:.3f}  "
                  f"d1 {r['delta1'] * 100:.1f}%  over {r['images']} images")
            if r.get("output_form_conflict"):
                print(f"  !! {r['output_form_conflict']}")
            for alt in ("scale", "scale_shift"):
                d = r.get(f"diag_{alt}")
                if d:
                    fit = d.get("fit_median", {})
                    extra = " ".join(f"{k}={v:.4g}" for k, v in fit.items())
                    print(f"    if {alt:12} were allowed: AbsRel "
                          f"{d['abs_rel'] * 100:.2f}%  d1 {d['delta1'] * 100:.1f}%"
                          f"   median {extra}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for r in results:
        with open(os.path.join(OUT_DIR, f"{r['model']}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(r, f, indent=2)
    out = os.path.join(ROOT, "reports", "gt.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(results, manifest))
    print(f"\nwrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
