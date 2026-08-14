"""Where does one model's time actually go?

    python tools/profile_model.py depth_anything_v2
    python tools/profile_model.py depth_anything_v2 --iterations 50 --top 25

Runs the engine the benchmark measured, on the exact input the benchmark used,
with TensorRT's layer profiler attached, and reports the time per layer
alongside the engine's own structure.

Uses reports/inputs/<model>.npy and every declared
reports/inputs/<model>__<binding>.npy -- the recorded tensors that produced the
benchmark number -- so the profile describes the same complete computation.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from core import bench, build_conditions, spec as spec_mod  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(ROOT, "reports", "inputs")


def saved_inputs(model, model_spec, inputs_dir=INPUTS):
    """Load every recorded engine input in declared binding order.

    Most models have only the image. TR2M also has a CLIP text embedding;
    profiling it with an uninitialised second binding measures a computation
    that the benchmark never ran. The spec is the ordering contract and the
    recorded arrays are the values the benchmark actually used.
    """
    declared = [model_spec["input"]] + list(model_spec.get("extra_inputs", []))
    feeds = []
    for i, item in enumerate(declared):
        suffix = "" if i == 0 else "__" + item["name"]
        path = os.path.join(inputs_dir, f"{model}{suffix}.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{model}: no saved input for binding {item['name']!r} at {path}")
        feeds.append((item["name"], np.ascontiguousarray(np.load(path))))
    return feeds


def bind_saved_inputs(slots, feeds):
    """Copy all declared feeds into TensorRT host buffers, without omissions."""
    if len(slots) != len(feeds):
        names = ", ".join(name for name, _ in feeds)
        raise ValueError(
            f"engine has {len(slots)} input bindings but the spec/record has "
            f"{len(feeds)} ({names})")
    for slot, (name, value) in zip(slots, feeds):
        flat = np.asarray(value, dtype=slot.host.dtype).ravel()
        if flat.size != slot.host.size:
            raise ValueError(
                f"binding {name!r} has {slot.host.size} elements but the "
                f"recorded input has {flat.size}")
        np.copyto(slot.host, flat, casting="safe")


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

    # --engine points at an engine the reports do not know about. That is how
    # two builds of one model get compared -- an older encoder, a different
    # simplification setting -- without a run overwriting the recorded one.
    engine_path = args.engine or run_rec.get("engine_path") or ""
    if not os.path.isfile(engine_path):
        print(f"{model}: engine not on this machine ({engine_path})")
        return 1

    try:
        feeds = saved_inputs(model, spec_mod.load(model))
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return 1
    feed = feeds[0][1]

    from core import common
    from core import common_runtime
    from core import profile as prof

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    bind_saved_inputs(inputs, feeds)

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

    env = bench.collect_env()
    payload = {
        "model": model,
        "variant": args.variant,
        "precision": run_rec.get("precision"),
        # Which kernels an engine contains depends on the GPU clock during its
        # build, so a profile is only readable next to the clock it was taken
        # under. Nine profiles had to be thrown away for want of this field.
        "clock_mhz": env.get("clock_mhz"),
        "clock_max_mhz": env.get("clock_max_mhz"),
        "engine_path": engine_path,
        # So compare.py can prove this profile describes the engine the
        # benchmark measured, rather than assuming it from the model name.
        **build_conditions.stamp(engine_path),
        "input_shape": list(feed.shape),
        "extra_inputs": [
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in feeds[1:]
        ],
        "iterations": args.iterations,
        "summary": summary,
        "suspects": susp,
        "layers": rows,
        "engine_layers": layers,
    }
    path = prof.save(model + (f"_{args.tag}" if args.tag else ""), payload)
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
    ap.add_argument("--engine", default=None,
                    help="profile this engine instead of the recorded one")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file, to keep two profiles apart")
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
