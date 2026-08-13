"""Build one model's graph under different builder settings and time each.

    python tune_build.py unik3d --opt-level 3 5
    python tune_build.py depth_anything_v2 --opt-level 0 1 2 3 4 5
    python tune_build.py unik3d --workspace 2 4 8

Builder optimisation level is how hard TensorRT searches for kernels. It does
not change what the engine computes in the mathematical sense, only how long
the build takes and which tactics are considered -- but it still needs an
accuracy check afterwards, because different kernels round differently. On
unik3d, level 5 moved a fifth of the runtime out of fp32 layers into fp16 ones,
which is exactly the kind of change that shows up in the last digits.

**Pin the GPU clock before using this, or the numbers below mean nothing.**
TensorRT chooses kernels by timing candidates on the GPU during the build, so
on a card whose clock swings between idle and boost the choice is partly noise.
Four builds of unik3d from the same ONNX with the same options gave 19.04 /
13.35 / 8.41 / 14.49 ms; three builds with the clock pinned at 1800 MHz gave
9.66 / 9.62 / 9.68. A sweep run unpinned compares coin flips, and that is
exactly how this tool first reported a 2 ms win for opt_level 5 that did not
survive being applied. `nvidia-smi -lgc 1800,1800`, and `-rgc` to release.

Why it still earns a tool: with the clock pinned, a difference between two
settings is a difference between two settings.

Engines land in engine/tune/ and nothing here writes reports/bench. The
recorded measurement stays the one the default build produced, so a tuning
experiment cannot quietly become the published number.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from core import bench, profile as prof, spec as spec_mod  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def measure_engine(engine_path, feed, iterations, warmup, want_profile):
    import tensorrt as trt

    import common
    import common_runtime

    with open(engine_path, "rb") as f, \
            trt.Runtime(trt.Logger(trt.Logger.WARNING)) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    np.copyto(inputs[0].host, feed.ravel().view(inputs[0].host.dtype))

    try:
        _, samples = bench.measure(
            lambda: common_runtime.do_inference(context, engine, bindings,
                                                inputs, outputs, stream),
            warmup=warmup, iterations=iterations)
        summary = None
        type_shares = {}
        if want_profile:
            timer = prof.LayerTimer()
            context.profiler = timer
            for _ in range(10):
                common_runtime.do_inference(context, engine, bindings, inputs,
                                            outputs, stream)
            rows = timer.rows(10)
            layers = prof.inspect(engine)
            summary = prof.summarize(rows, layers)
            layer_types = {l.get("Name"): l.get("LayerType", "?")
                           for l in layers if l.get("Name")}
            for row in rows:
                kind = layer_types.get(row["name"], "?")
                type_shares[kind] = type_shares.get(kind, 0.0) + row["share"]
    finally:
        common.free_buffers(inputs, outputs, stream)

    b = bench.Bench(model="tune", samples_ms=samples)
    return b.stats(), summary, type_shares


def _safe_tag(value):
    """Keep experiment labels usable as filenames on both Windows and Linux."""
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not tag:
        raise ValueError("experiment tag is empty after filename sanitization")
    return tag


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--opt-level", type=int, nargs="*", default=[None])
    ap.add_argument("--workspace", type=int, nargs="*", default=[None])
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--onnx", default=None,
                    help="build this ONNX instead of the model's recorded graph")
    ap.add_argument("--tag", default=None,
                    help="filename label for a custom ONNX experiment")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent builds per setting (separate timing caches)")
    ap.add_argument("--out", default=None,
                    help="optional JSON file for all measurements and profile shares")
    args = ap.parse_args()

    if args.repeats < 1:
        ap.error("--repeats must be at least 1")

    specs = spec_mod.load_all()
    if args.model not in specs:
        print(f"no spec for {args.model}")
        return 1

    run_rec = None
    for r in bench.load_all(os.path.join(ROOT, "reports", "bench")):
        if r["model"] == args.model and r.get("variant", "single") == "single":
            run_rec = r
    if not run_rec:
        print(f"{args.model}: not measured yet - build it first")
        return 1

    from package_artifacts import onnx_beside
    if args.onnx:
        onnx_path = os.path.abspath(args.onnx)
    else:
        onnx_path = onnx_beside(run_rec.get("engine_path") or "")
    if not onnx_path or not os.path.isfile(onnx_path):
        print(f"{args.model}: source ONNX not found: {onnx_path or run_rec.get('engine_path')}")
        return 1

    feed = np.load(os.path.join(ROOT, "reports", "inputs", f"{args.model}.npy"))
    precision = run_rec.get("precision", "fp16")
    baseline = (run_rec.get("stats") or {}).get("mean_ms")

    out_dir = os.path.join(os.path.dirname(run_rec["engine_path"]), "tune")
    os.makedirs(out_dir, exist_ok=True)
    if args.onnx:
        label = _safe_tag(args.tag or os.path.splitext(os.path.basename(onnx_path))[0])
        stem = f"{label}_{precision}"
    else:
        stem = os.path.splitext(os.path.basename(run_rec["engine_path"]))[0]

    import common
    print(f"\n{args.model}  ({precision}, recorded build {baseline} ms)")
    print(f"source ONNX  {onnx_path}")
    print(f"{'opt':>4}{'ws':>4}{'run':>5}{'mean ms':>10}{'p50':>9}"
          f"{'fp32':>8}{'reformat':>10}{'slice':>8}  vs recorded")
    print("-" * 84)

    results = []

    for opt in args.opt_level:
        for ws in args.workspace:
            setting = f"opt{opt if opt is not None else 'd'}_ws{ws if ws is not None else 'd'}"
            for repeat in range(1, args.repeats + 1):
                suffix = f"_{setting}"
                if args.repeats > 1 or args.onnx:
                    suffix += f"_r{repeat}"
                engine_path = os.path.join(out_dir, f"{stem}{suffix}.engine")
                kw = {}
                if opt is not None:
                    kw["opt_level"] = opt
                if ws is not None:
                    kw["workspace_gib"] = ws
                common.get_engine(onnx_path, engine_path, precision, **kw)
                stats, summary, type_shares = measure_engine(
                    engine_path, feed, args.iterations, args.warmup,
                    not args.no_profile)
                delta = ""
                if baseline:
                    pct = (stats["mean_ms"] / baseline - 1.0) * 100.0
                    delta = f"{pct:+.1f}%"
                fp32_share = summary["fp32_output_share"] if summary else None
                fp32 = f"{fp32_share * 100:.1f}%" if fp32_share is not None else "-"
                reformat = type_shares.get("Reformat")
                slice_share = type_shares.get("Slice")
                refmt_text = f"{reformat * 100:.1f}%" if reformat is not None else "-"
                slice_text = f"{slice_share * 100:.1f}%" if slice_share is not None else "-"
                print(f"{str(opt):>4}{str(ws):>4}{repeat:5d}{stats['mean_ms']:10.2f}"
                      f"{stats['p50_ms']:9.2f}{fp32:>8}{refmt_text:>10}"
                      f"{slice_text:>8}  {delta}")
                results.append({
                    "model": args.model,
                    "onnx_path": onnx_path,
                    "engine_path": engine_path,
                    "precision": precision,
                    "opt_level": opt,
                    "workspace_gib": ws,
                    "repeat": repeat,
                    "stats": stats,
                    "environment": bench.collect_env(),
                    "profile_summary": summary,
                    "layer_type_time_shares": type_shares,
                })

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nMeasurements written to {out_path}")

    print("\nNothing was written to reports/bench. Engines are under "
          f"{os.path.relpath(out_dir, ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
