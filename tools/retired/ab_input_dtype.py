"""Does moving preprocessing into the engine pay for itself?

    python tools/retired/ab_input_dtype.py depth_anything_v2
    python tools/retired/ab_input_dtype.py --all
    python tools/retired/ab_input_dtype.py depth_pro --iterations 200

Today the caller hands the engine float32 NCHW: divide by 255, subtract the
mean, divide by the standard deviation, transpose, copy. Taking uint8 NHWC
instead cuts the host-to-device copy to a quarter, and adds a Cast, two Divs, a
Sub and a Transpose to the graph. Whether that is a win depends on whether
TensorRT fuses the preamble into the first convolution, which it does not
promise to do.

One total cannot answer that, so this reports the phases separately -- host to
device, kernels, device to host, and what is left over. A 3% difference in the
total tells you nothing about why; a copy that halved while compute grew tells
you everything.

Both engines are built from the same ONNX body: the uint8 variant is the
baseline graph with five nodes prepended, so nothing but the preamble differs.

Correctness is checked in the same run. The two engines are fed equivalent
data -- one the uint8 picture, the other that picture normalized on the host --
and their outputs compared. A faster engine that computes something else is
not a faster engine.

Writes reports/uint8_ab/<model>.json and prints a table. Adopt nothing
automatically: the plan is explicit that this applies per profile where it
wins, not everywhere.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from core import bench, spec as spec_mod  # noqa: E402
from core.onnx_tools import add_uint8_input  # noqa: E402
from tools.package_artifacts import onnx_beside  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "reports", "uint8_ab")


def host_preprocess(img_u8, norm):
    """What the model scripts do today, in float32 throughout.

    Deliberately not `img / 255.0`: numpy promotes a uint8 array divided by a
    Python float to float64, which is how three graphs ended up carrying
    float64 (see docs/model_contracts.md). It makes no difference to the
    result here, but the habit is what caused the problem.
    """
    f = img_u8.astype(np.float32) / np.float32(norm.get("scale", 255.0))
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)
    f = (f - mean) / std
    rank = f.ndim
    perm = list(range(rank - 3)) + [rank - 1, rank - 3, rank - 2]
    return np.ascontiguousarray(f.transpose(perm))


def deterministic_image(shape_nhwc, seed=0):
    """Timing does not depend on pixel values, but reproducibility does: the
    output comparison has to mean the same thing on a second run."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape_nhwc, dtype=np.uint8)


def run_engine(engine_path, feed, iterations, warmup):
    """Build/load, run the staged loop, return (outputs, stats-ready pieces)."""
    import tensorrt as trt

    from core import common
    from core import common_runtime
    from core.common_runtime import StageTimer

    # Deserialize directly rather than through get_engine: the engine was just
    # built, and get_engine would compare fingerprints and rebuild on any
    # mismatch, which in a timing comparison would silently replace one of the
    # two things being compared.
    with open(engine_path, "rb") as f, \
            trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"could not deserialize {engine_path}")

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    np.copyto(inputs[0].host, feed.ravel().view(inputs[0].host.dtype))

    timer = StageTimer()
    try:
        def once():
            return common_runtime.do_inference(context, engine, bindings,
                                               inputs, outputs, stream,
                                               timer=timer)

        out, samples, stages = bench.measure_staged(
            once, lambda: timer.last, warmup=warmup, iterations=iterations,
            sync=lambda: None)   # do_inference already synchronizes the stream
        # Copy before the finally block frees the buffers. do_inference returns
        # views into pinned host memory, not copies, so reading them after
        # free_buffers is a use-after-free -- which on Windows is not an
        # exception but a silent process death, no traceback, exit code lost in
        # the pipeline. That is what this cost to find.
        out = [np.array(o) for o in out]
    finally:
        timer.free()
        common.free_buffers(inputs, outputs, stream)

    return out, samples, stages


def compare_outputs(a, b):
    """Relative difference per output, plus correlation.

    Relative difference alone misleads on a tensor centred near zero -- dav3's
    sky mask has |mean| 6.6e-06, where a rounding difference reads as 8%. The
    correlation says whether the same picture came out.
    """
    rows = []
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = x.astype(np.float64).ravel(), y.astype(np.float64).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        denom = np.abs(x).mean() or 1.0
        corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 and x.std() else 1.0
        rows.append({"output": i, "rel": float(np.abs(x - y).mean() / denom),
                     "corr": corr, "ref_abs_mean": float(np.abs(x).mean())})
    return rows


def builder_options(engine_path):
    """The builder settings the baseline engine was built with.

    They differ per model -- vggt asks for workspace_gib=4, the rest take the
    default -- and they are not recorded in spec.json or in the benchmark. They
    are in the .fingerprint file get_engine writes beside every engine, which
    exists precisely so a stale engine is not reused, and it turns out to be
    the only machine-readable record of how each one was built.

    Building the variant with different settings would put a second difference
    into a two-way comparison.
    """
    opts = {"workspace_gib": 2, "opt_level": None}
    fp = os.path.splitext(engine_path)[0] + ".fingerprint"
    if not os.path.isfile(fp):
        return opts
    for line in open(fp, encoding="utf-8"):
        key, _, value = line.strip().partition("=")
        if key == "workspace":
            opts["workspace_gib"] = int(float(value))
        elif key == "opt_level" and value not in ("None", ""):
            opts["opt_level"] = int(value)
    return opts


def one(name, spec, args):
    norm = (spec.get("input") or {}).get("normalize")
    if not norm:
        print(f"{name:20} no input.normalize in spec.json - cannot build the "
              f"preamble without the constants the host uses")
        return None

    runs = {r["model"]: r for r in bench.load_all(os.path.join(ROOT, "reports", "bench"))}
    run = runs.get(name)
    if not run:
        print(f"{name:20} not measured - build it first")
        return None
    base_onnx = onnx_beside(run.get("engine_path") or "")
    if not base_onnx or not os.path.isfile(base_onnx):
        print(f"{name:20} baseline ONNX not on this machine")
        return None

    precision = run.get("precision", "fp16")
    u8_onnx = os.path.splitext(base_onnx)[0] + "_u8.onnx"
    if not os.path.isfile(u8_onnx) or args.rebuild:
        add_uint8_input(base_onnx, u8_onnx, norm["mean"], norm["std"],
                        scale=norm.get("scale", 255.0))

    from core import common
    eng_dir = os.path.dirname(run["engine_path"])
    a_engine = run["engine_path"]
    b_engine = os.path.join(
        eng_dir, os.path.splitext(os.path.basename(u8_onnx))[0] + f"_{precision}.engine")

    opts = builder_options(a_engine)
    print(f"\n=== {name} | {precision} | {opts}")
    if not os.path.isfile(a_engine):
        print(f"{name:20} baseline engine not on this machine")
        return None
    # The baseline is used exactly as measured, never rebuilt. get_engine would
    # rebuild it on any fingerprint difference, and that is not a hypothetical:
    # vggt's onnx2trt.py passes workspace_gib=4, so the default would have
    # replaced a 4.7 GB engine mid-comparison and compared the variant against
    # something the reports never measured.
    common.get_engine(u8_onnx, b_engine, precision, **opts)

    import onnx
    dims = [d.dim_value for d in
            onnx.load(u8_onnx, load_external_data=False).graph.input[0]
            .type.tensor_type.shape.dim]
    img = deterministic_image(dims)
    nchw = host_preprocess(img, norm)
    print(f"    input {tuple(img.shape)} uint8 -> {tuple(nchw.shape)} float32",
          flush=True)

    print("    measuring float32 NCHW", flush=True)
    a_out, a_samples, a_stages = run_engine(a_engine, nchw, args.iterations, args.warmup)
    print("    measuring uint8 NHWC", flush=True)
    b_out, b_samples, b_stages = run_engine(b_engine, img, args.iterations, args.warmup)

    a = bench.Bench(model=name, samples_ms=a_samples, stage_samples_ms=a_stages)
    b = bench.Bench(model=name, samples_ms=b_samples, stage_samples_ms=b_stages)
    sa, sb = a.stats(), b.stats()

    result = {
        "model": name, "precision": precision,
        "input_bytes": {"float32_nchw": int(nchw.nbytes), "uint8_nhwc": int(img.nbytes)},
        "float32_nchw": sa, "uint8_nhwc": sb,
        "delta_ms": round(sb["mean_ms"] - sa["mean_ms"], 4),
        "delta_pct": round((sb["mean_ms"] / sa["mean_ms"] - 1.0) * 100.0, 2),
        "outputs": compare_outputs(a_out, b_out),
        "engines": {"float32_nchw": a_engine, "uint8_nhwc": b_engine},
    }

    print(f"\n{name} | {precision}  ({args.iterations} iterations)")
    print(f"  {'':16} {'total':>9} {'h2d':>8} {'compute':>9} {'d2h':>8} {'host':>8}")
    for label, s in (("float32 NCHW", sa), ("uint8 NHWC", sb)):
        print(f"  {label:16} {s['mean_ms']:9.3f} {s.get('h2d_ms', 0):8.3f} "
              f"{s.get('compute_ms', 0):9.3f} {s.get('d2h_ms', 0):8.3f} "
              f"{s.get('host_overhead_ms', 0):8.3f}")
    print(f"  {'delta':16} {result['delta_ms']:+9.3f} "
          f"({result['delta_pct']:+.2f}%)   "
          f"copy {nchw.nbytes / 1e6:.2f} MB -> {img.nbytes / 1e6:.2f} MB")
    for row in result["outputs"]:
        print(f"  output {row['output']}: rel {row['rel'] * 100:.4f}%  "
              f"corr {row['corr']:.6f}  ref |mean| {row['ref_abs_mean']:.3e}")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  -> {os.path.relpath(path, ROOT)}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*")
    ap.add_argument("--all", action="store_true",
                    help="every model whose spec records input.normalize")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate the uint8 ONNX even if it is there")
    args = ap.parse_args()

    specs = spec_mod.load_all()
    wanted = ([n for n, s in sorted(specs.items())
               if (s.get("input") or {}).get("normalize")]
              if args.all else args.models)
    if not wanted:
        print("name a model or pass --all")
        return 2

    results = [one(n, specs[n], args) for n in wanted if n in specs]
    results = [r for r in results if r]
    if len(results) > 1:
        print("\n" + "=" * 62)
        for r in results:
            verdict = "uint8 wins" if r["delta_pct"] < 0 else "keep float32"
            print(f"{r['model']:20} {r['delta_pct']:+7.2f}%   {verdict}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
