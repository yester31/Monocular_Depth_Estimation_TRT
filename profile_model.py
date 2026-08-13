"""Where does one model's time actually go?

    python profile_model.py depth_anything_v2
    python profile_model.py depth_anything_v2 --iterations 50 --top 25

Runs the engine the benchmark measured, on the exact input the benchmark used,
with TensorRT's layer profiler attached, and reports the time per layer
alongside the engine's own structure.

Uses reports/inputs/<model>.npy -- the byte-identical tensor that produced the
recorded number -- so the profile describes the run in the report and not some
other run.

Two things this is for. Finding where the time is, obviously. And finding
layers that cost time while doing no arithmetic: a Reformat or a Shuffle is
TensorRT reconciling two neighbours that disagree about layout or precision,
and a pile of those is what a graph problem looks like from the inside. That is
the shape the float64 defect took on unidepth_v2, where it cost 20%.

Nothing is changed or rebuilt here.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from core import bench, profile as prof, spec as spec_mod  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_engine(path):
    import tensorrt as trt
    with open(path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"could not deserialize {path}")
    return engine


def run(model, args):
    runs = {}
    for r in bench.load_all(os.path.join(ROOT, "reports", "bench")):
        # Keep the plain build, not a variant, unless one was asked for.
        if r["model"] == model and r.get("variant", "single") == args.variant:
            runs[model] = r
    run_rec = runs.get(model)
    if not run_rec:
        print(f"{model}: no benchmark record for variant {args.variant!r} - "
              f"build it first")
        return 1

    engine_path = run_rec.get("engine_path") or ""
    if not os.path.isfile(engine_path):
        print(f"{model}: engine not on this machine ({engine_path})")
        return 1

    input_path = os.path.join(ROOT, "reports", "inputs", f"{model}.npy")
    if not os.path.isfile(input_path):
        print(f"{model}: no saved input at {input_path}")
        return 1
    feed = np.load(input_path)

    import common
    import common_runtime

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    np.copyto(inputs[0].host, feed.ravel().view(inputs[0].host.dtype))

    timer = prof.LayerTimer()
    try:
        # Warm up without the profiler attached: the first executions include
        # one-off setup that would otherwise be charged to whichever layer
        # happened to run first.
        for _ in range(args.warmup):
            common_runtime.do_inference(context, engine, bindings, inputs,
                                        outputs, stream)
        context.profiler = timer
        for _ in range(args.iterations):
            common_runtime.do_inference(context, engine, bindings, inputs,
                                        outputs, stream)
        # Not `context.profiler = None`: TensorRT rejects a null profiler with
        # an API error on stderr, which reads like a failure in the middle of a
        # successful run. The context is discarded below anyway.

        rows = timer.rows(args.iterations)
        layers = prof.inspect(engine)
    finally:
        common.free_buffers(inputs, outputs, stream)

    stats = run_rec.get("stats") or {}
    summary = prof.summarize(rows, layers, mean_ms=stats.get("mean_ms"))
    susp = prof.suspects(rows, layers)

    print(prof.render(model, rows, summary, susp, top=args.top))

    payload = {
        "model": model,
        "variant": args.variant,
        "precision": run_rec.get("precision"),
        "engine_path": engine_path,
        "input_shape": list(feed.shape),
        "iterations": args.iterations,
        "summary": summary,
        "suspects": susp,
        "layers": rows,
        "engine_layers": layers,
    }
    path = prof.save(model, payload)
    print(f"\n-> {os.path.relpath(path, ROOT)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--variant", default="single")
    args = ap.parse_args()

    specs = spec_mod.load_all()
    rc = 0
    for m in args.models:
        if m not in specs:
            print(f"no spec for {m}")
            rc = 1
            continue
        rc |= run(m, args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
