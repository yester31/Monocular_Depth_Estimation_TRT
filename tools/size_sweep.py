"""Does the input size a model is given change how accurate it is?

    python tools/size_sweep.py depth_anything_v2 --sizes 518x518 672x896 \
        --manifest data/eval/diode_indoors.json --root C:/Users/soy/data/diode/val

D3. The old sensitivity table in docs/model_contracts.md answers a different
question -- it compares each model against *its own* output at another size, so
reproducing its baselines means reproducing that PyTorch path. Ground truth
needs no reconstruction: point the model at real measured depth at each size
and read off what changes.

**Nothing here writes to reports/bench or reports/inputs.** That rule is the
whole reason this file exists. Building a second engine through a model's own
onnx2trt.py leaves two benchmark records and an overwritten reference tensor,
and then nothing knows which engine the published row describes -- which is
exactly what happened on the first attempt at D3. Engines land in
engine/size/, results in reports/tune/size_<model>.json.

The export is the awkward part: a model's input size lives in its
onnx_export.py as a literal, and the size has to change without editing the
file everyone else uses. So the script is copied beside itself with the size
substituted, run, and deleted. Copied *into the model's directory* rather than
a temp one, because these scripts resolve their clones and weights relative to
themselves.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core import bench, gt, sizes as size_lib, spec as spec_mod  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_size(text):
    try:
        return size_lib.parse(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def export_at(model, h, w, env_prefix, dry_run=False):
    """Export this model's ONNX at (h, w). Returns the path, or raises."""
    model_dir = os.path.join(ROOT, "models", model)
    src_path = os.path.join(model_dir, "onnx_export.py")
    with open(src_path, encoding="utf-8") as f:
        source = f.read()
    can, why = size_lib.sweepable(model)
    if not can:
        raise RuntimeError(f"{model}: input size cannot be swept -- {why}")
    rewritten = size_lib.rewrite(source, h, w)
    if rewritten is None:
        raise RuntimeError(
            f"{model}: onnx_export.py does not set input_h/input_w in a shape "
            f"core/sizes.py understands; add the spelling to PATTERNS rather "
            f"than exporting it unchanged")

    tmp_name = f"_size_sweep_{h}x{w}.py"
    tmp_path = os.path.join(model_dir, tmp_name)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(rewritten)
    try:
        cmd = [env_prefix, "run", "--no-capture-output", "-p",
               spec_env_prefix(model), "python", tmp_name]
        print(f"[MDET] exporting {model} at {h}x{w}")
        if dry_run:
            print("       " + " ".join(cmd))
            return None
        env = dict(os.environ, PYTHONUTF8="1")
        r = subprocess.run(cmd, cwd=model_dir, env=env)
        if r.returncode:
            raise RuntimeError(f"{model}: export at {h}x{w} exited {r.returncode}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return newest_onnx(model_dir, h, w)


def newest_onnx(model_dir, h, w):
    """The ONNX just written for this size, found by its name carrying it."""
    onnx_dir = os.path.join(model_dir, "onnx")
    hits = [os.path.join(onnx_dir, n) for n in os.listdir(onnx_dir)
            if f"{h}x{w}" in n and n.endswith(".onnx") and "_sim" not in n]
    if not hits:
        raise RuntimeError(f"no ONNX named for {h}x{w} under {onnx_dir}")
    return max(hits, key=os.path.getmtime)


def spec_env_prefix(model):
    envs, _ = conda_envs()
    want = spec_mod.load_all()[model]["env"]
    prefix = envs.get(want)
    if not prefix:
        raise RuntimeError(f"{model}: conda environment {want!r} not on this machine")
    return prefix


def conda_envs():
    from run import conda_envs as _envs
    return _envs()


def measure_and_score(model, onnx_path, h, w, manifest, root, limit):
    """Build an engine for this ONNX, time it, and score it on the manifest."""
    from core import common
    import evaluate_gt as ev

    engine_dir = os.path.join(ROOT, "models", model, "engine", "size")
    os.makedirs(engine_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    engine_path = os.path.join(engine_dir, f"{stem}_fp16.engine")

    engine = common.get_engine(onnx_path, engine_path, "fp16")
    sp = spec_mod.load_all()[model]
    policy = gt.policy_for(sp.get("depth_scale"))
    pred_is_inverse = sp.get("output_form") == "inverse_depth"

    per_sample = []
    with engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        samples = manifest["samples"][:limit] if limit else manifest["samples"]

        # Time it on the first image, so the speed and the accuracy describe
        # the same engine on the same input.
        first = cv2.imread(os.path.join(root, samples[0]["rgb"]))
        inputs[0].host = ev.ADAPTERS[model]["pre"](first, h, w)
        _, times = bench.measure(
            lambda: common.do_inference(context, engine=engine, bindings=bindings,
                                        inputs=inputs, outputs=outputs, stream=stream),
            warmup=10, iterations=50)
        stats = bench.Bench(model=model, samples_ms=times).stats()

        for s in samples:
            bgr = cv2.imread(os.path.join(root, s["rgb"]))
            if bgr is None:
                continue
            oh, ow = bgr.shape[:2]
            inputs[0].host = ev.ADAPTERS[model]["pre"](bgr, h, w)
            raw = common.do_inference(context, engine=engine, bindings=bindings,
                                      inputs=inputs, outputs=outputs, stream=stream)
            pred = ev.ADAPTERS[model]["depth"](raw, h, w, oh, ow, manifest)

            g = np.load(os.path.join(root, s["depth"]))
            m = np.load(os.path.join(root, s["mask"])).astype(bool)
            g = g[..., 0] if g.ndim == 3 else g
            m = m[..., 0] if m.ndim == 3 else m
            m = m & (g > 0)
            try:
                aligned, _ = gt.align(pred, g, m, policy, pred_is_inverse)
            except ValueError:
                continue
            r = gt.metrics(aligned, g, m, max_depth=manifest.get("max_depth"))
            if not r.get("n"):
                continue
            # A metric model scored without alignment mixes two things: whether
            # the shape of the depth changed, and whether the scale did. At a
            # different input size both can move, and AbsRel alone cannot say
            # which -- so the scale-aligned score comes along, as a diagnostic.
            if policy == "none":
                try:
                    a2, fit = gt.align(pred, g, m, "scale", pred_is_inverse)
                    d = gt.metrics(a2, g, m, max_depth=manifest.get("max_depth"))
                    r["scaled_abs_rel"] = d["abs_rel"]
                    r["scaled_delta1"] = d["delta1"]
                    r["fitted_scale"] = fit.get("scale")
                except ValueError:
                    pass
            per_sample.append(r)
        common.free_buffers(inputs, outputs, stream)

    if not per_sample:
        raise RuntimeError(f"{model} at {h}x{w}: nothing scored")
    agg = {k: float(np.mean([r[k] for r in per_sample]))
           for k in ("abs_rel", "rmse", "delta1")}
    for k in ("scaled_abs_rel", "scaled_delta1"):
        vals = [r[k] for r in per_sample if k in r]
        if vals:
            agg[k] = float(np.mean(vals))
    scales = [r["fitted_scale"] for r in per_sample if r.get("fitted_scale")]
    if scales:
        agg["fitted_scale_median"] = float(np.median(scales))
    agg.update(size=f"{h}x{w}", images=len(per_sample), alignment=policy,
               mean_ms=stats["mean_ms"], p50_ms=stats["p50_ms"],
               engine_path=engine_path, onnx_path=onnx_path)
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--sizes", nargs="+", required=True, type=parse_size,
                    metavar="HxW")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    specs = spec_mod.load_all()
    if args.model not in specs:
        print(f"no spec for {args.model}")
        return 1
    import evaluate_gt as ev
    if args.model not in ev.ADAPTERS:
        print(f"{args.model}: no ground-truth adapter, so a size sweep has "
              f"nothing to score")
        return 1

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    _, exe = conda_envs()

    results = []
    for h, w in args.sizes:
        try:
            onnx_path = export_at(args.model, h, w, exe, args.dry_run)
            if args.dry_run:
                continue
            results.append(measure_and_score(args.model, onnx_path, h, w,
                                             manifest, args.root, args.limit))
        except Exception as e:                                # noqa: BLE001
            results.append({"size": f"{h}x{w}", "error": f"{type(e).__name__}: {e}"})
            print(f"[MDET] {h}x{w} failed: {e}")
            continue
        r = results[-1]
        extra = ""
        if "scaled_abs_rel" in r:
            extra = (f"   scaled {r['scaled_abs_rel'] * 100:5.2f}% / "
                     f"d1 {r['scaled_delta1'] * 100:5.1f}%   "
                     f"scale {r.get('fitted_scale_median', float('nan')):.3f}")
        print(f"[MDET] {r['size']:>10}  {r['mean_ms']:7.2f} ms   "
              f"AbsRel {r['abs_rel'] * 100:6.2f}%   d1 {r['delta1'] * 100:5.1f}%{extra}")

    if args.dry_run:
        return 0
    out_dir = os.path.join(ROOT, "reports", "tune")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"size_{args.model}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "manifest": args.manifest,
                   "results": results}, f, indent=2)
    print(f"\nwrote {os.path.relpath(out, ROOT)}. "
          f"Nothing was written to reports/bench or reports/inputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
