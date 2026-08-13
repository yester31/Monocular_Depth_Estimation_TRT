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

Why it earns a tool: unik3d on a backbone with a third of the weights came out
slower than the larger one, and the profile showed why -- the transformer body
lost its fp16 tactics and its 18 fusion layers, so `gemm` grew by 2 ms on a
smaller model. That is a search outcome, and the search depth is a dial.

Engines land in engine/tune/ and nothing here writes reports/bench. The
recorded measurement stays the one the default build produced, so a tuning
experiment cannot quietly become the published number.
"""

import argparse
import os
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
        share = None
        if want_profile:
            timer = prof.LayerTimer()
            context.profiler = timer
            for _ in range(10):
                common_runtime.do_inference(context, engine, bindings, inputs,
                                            outputs, stream)
            rows = timer.rows(10)
            layers = prof.inspect(engine)
            share = prof.summarize(rows, layers)["fp32_output_share"]
    finally:
        common.free_buffers(inputs, outputs, stream)

    b = bench.Bench(model="tune", samples_ms=samples)
    return b.stats(), share


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--opt-level", type=int, nargs="*", default=[None])
    ap.add_argument("--workspace", type=int, nargs="*", default=[None])
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()

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
    onnx_path = onnx_beside(run_rec.get("engine_path") or "")
    if not onnx_path or not os.path.isfile(onnx_path):
        print(f"{args.model}: source ONNX not found for {run_rec.get('engine_path')}")
        return 1

    feed = np.load(os.path.join(ROOT, "reports", "inputs", f"{args.model}.npy"))
    precision = run_rec.get("precision", "fp16")
    baseline = (run_rec.get("stats") or {}).get("mean_ms")

    out_dir = os.path.join(os.path.dirname(run_rec["engine_path"]), "tune")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(run_rec["engine_path"]))[0]

    import common
    print(f"\n{args.model}  ({precision}, recorded build {baseline} ms)")
    print(f"{'opt':>4}{'ws':>4}{'mean ms':>10}{'p50':>9}{'fp32':>8}  vs recorded")
    print("-" * 52)

    for opt in args.opt_level:
        for ws in args.workspace:
            tag = f"opt{opt if opt is not None else 'd'}_ws{ws if ws is not None else 'd'}"
            engine_path = os.path.join(out_dir, f"{stem}_{tag}.engine")
            kw = {}
            if opt is not None:
                kw["opt_level"] = opt
            if ws is not None:
                kw["workspace_gib"] = ws
            common.get_engine(onnx_path, engine_path, precision, **kw)
            stats, share = measure_engine(engine_path, feed, args.iterations,
                                          args.warmup, not args.no_profile)
            delta = ""
            if baseline:
                pct = (stats["mean_ms"] / baseline - 1.0) * 100.0
                delta = f"{pct:+.1f}%"
            fp32 = f"{share * 100:.1f}%" if share is not None else "-"
            print(f"{str(opt):>4}{str(ws):>4}{stats['mean_ms']:10.2f}"
                  f"{stats['p50_ms']:9.2f}{fp32:>8}  {delta}")

    print("\nNothing was written to reports/bench. Engines are under "
          f"{os.path.relpath(out_dir, ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
