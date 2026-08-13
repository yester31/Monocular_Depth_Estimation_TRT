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
    policy = gt.policy_for(spec_mod.load_all()[model].get("depth_scale"))
    if policy is None:
        return {"model": model, "skip": "no alignment policy for this output contract"}

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
            try:
                aligned, fit = gt.align(pred, g, m, policy)
            except ValueError as e:
                per_sample.append({"id": s["id"], "skip": str(e)})
                continue
            r = gt.metrics(aligned, g, m, max_depth=manifest.get("max_depth"))
            r["id"] = s["id"]
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
                        a2, f2 = gt.align(pred, g, m, alt)
                    except ValueError:
                        continue
                    r[f"diag_{alt}"] = gt.metrics(
                        a2, g, m, max_depth=manifest.get("max_depth"))
                    r[f"diag_{alt}_fit"] = f2
            per_sample.append(r)
            if fit:
                fits.append(fit)
        common.free_buffers(inputs, outputs, stream)

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
