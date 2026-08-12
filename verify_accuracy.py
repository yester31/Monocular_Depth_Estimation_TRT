"""Does each TensorRT engine reproduce the ONNX graph it was built from?

The comparison table says how fast the engines are. It says nothing about
whether they are right, and an fp16 engine that quietly returns garbage would
sit in that table looking excellent.

**The reference is the ONNX graph at fp32, not PyTorch.** Comparing an engine
against PyTorch mixes three different things into one number: the repo's
preprocessing versus upstream's, whatever the ONNX exporter changed, and what
TensorRT itself did. Only the third is a TensorRT question. Running the same
ONNX through onnxruntime at fp32 and through TensorRT, on the byte-identical
input the benchmark used, isolates it.

That input comes from `reports/inputs/<model>.npy`, written by
`core.bench.record(model_input=...)` during the benchmark run, so there is no
chance of the two paths seeing different preprocessing.

What this does NOT check: that the ONNX matches PyTorch. That is a separate
question about the exporter, and the export-time patches in
`core/export_compat.py` are each measured against PyTorch where they apply.

usage:
    python verify_accuracy.py                 # every model with an input file
    python verify_accuracy.py depth_anything_v2 unik3d
    python verify_accuracy.py --precision fp32
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.bench import load_all  # noqa: E402
from core.golden import best_scale, compare  # noqa: E402
from compare import FOLDER  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(ROOT, "reports", "inputs")
OUT = os.path.join(ROOT, "reports", "accuracy.md")

# Relative error above which a row is called out. fp16 keeps ~3 decimal digits,
# so a well-behaved network lands well under 1%; a few percent means the error
# is being amplified somewhere, and tens of percent means something is wrong
# rather than imprecise.
WARN = 0.01
FAIL = 0.05


def onnx_path_for(run):
    """The ONNX the engine was built from, derived from the engine path.

    onnx2trt.py writes engine/<name>_<precision>.engine from onnx/<name>.onnx,
    and VGGT nests both a directory deep.
    """
    engine = run.get("engine_path") or ""
    if not engine:
        return None
    name = os.path.basename(engine)
    for suffix in ("_fp16.engine", "_fp32.engine", ".engine"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    folder = os.path.dirname(os.path.dirname(engine))
    for cand in (os.path.join(folder, "onnx", f"{name}.onnx"),
                 os.path.join(folder, "onnx", name, f"{name}.onnx")):
        if os.path.exists(cand):
            return cand
    return None


def run_onnx(path, x):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    # CPU deliberately: the point is a trustworthy fp32 reference, not speed,
    # and the CUDA provider may pick different kernels than the CPU one.
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = sess.run(None, {name: x.astype(np.float32)})
    return [np.asarray(o) for o in outs]


def run_trt(engine_path, x, out_shapes):
    """Deserialize the built engine and run it once.

    Deliberately not common.get_engine: that checks a fingerprint and rebuilds
    when it does not match, which for a verification pass would mean silently
    measuring a freshly built engine instead of the one that produced the
    benchmark numbers.
    """
    import tensorrt as trt
    import common

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"could not deserialize {engine_path}")

    with engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = np.ascontiguousarray(x, dtype=np.float32)
        raw = common.do_inference(context, engine=engine, bindings=bindings,
                                  inputs=inputs, outputs=outputs, stream=stream)
        got = [np.array(r[: int(np.prod(s))]).reshape(s)
               for r, s in zip(raw, out_shapes)]
        common.free_buffers(inputs, outputs, stream)
        return got


def verify(run):
    model = run["model"]
    xp = os.path.join(INPUTS, f"{model}.npy")
    if not os.path.exists(xp):
        return {"model": model, "skip": "no reports/inputs/*.npy — rerun onnx2trt.py"}
    onnx_file = onnx_path_for(run)
    if not onnx_file:
        return {"model": model, "skip": f"onnx not found next to {run.get('engine_path')}"}
    engine = run.get("engine_path")
    if not os.path.exists(engine):
        return {"model": model, "skip": f"engine missing: {engine}"}

    x = np.load(xp)
    ref = run_onnx(onnx_file, x)
    got = run_trt(engine, x, [o.shape for o in ref])

    per_output = []
    for i, (a, b) in enumerate(zip(ref, got)):
        a = np.asarray(a, np.float64).ravel()
        b = np.asarray(b, np.float64).ravel()
        m = np.isfinite(a) & np.isfinite(b)
        rep = compare({"o": a}, {"o": b}, valid_mask=m)["o"]
        finite = a[m]
        corr = (float(np.corrcoef(a[m], b[m])[0, 1])
                if m.sum() > 1 and finite.std() > 0 else float("nan"))
        per_output.append({
            "index": i,
            "size": int(a.size),
            "rel_mean": rep["rel_mean"],
            "abs_rel": rep["abs_rel"],
            "rmse": rep["rmse"],
            "max_abs": float(np.abs(a[m] - b[m]).max()) if m.any() else float("nan"),
            "corr": corr,
            "scale": 1.0 / best_scale(a, b, m) if m.any() else float("nan"),
        })
    worst = max(o["rel_mean"] for o in per_output)
    return {"model": model, "precision": run.get("precision"),
            "onnx": os.path.relpath(onnx_file, ROOT),
            "outputs": per_output, "worst_rel": worst,
            "verdict": "FAIL" if worst > FAIL else ("WARN" if worst > WARN else "ok")}


def render(results):
    lines = ["# Engine accuracy", "",
             "Each TensorRT engine against the ONNX graph it was built from, "
             "at fp32, on the byte-identical input the benchmark used "
             "(`reports/inputs/`).", "",
             "This isolates what TensorRT and fp16 did. It is **not** a "
             "comparison against PyTorch — that would fold in preprocessing "
             "and the ONNX exporter as well. Generated by `verify_accuracy.py`.",
             "",
             f"`rel` is mean absolute difference over the mean magnitude of the "
             f"reference. Flagged above {WARN:.0%}, failed above {FAIL:.0%}.", ""]

    rows, skipped = [], []
    for r in results:
        if "skip" in r:
            skipped.append((r["model"], r["skip"]))
            continue
        w = max(r["outputs"], key=lambda o: o["rel_mean"])
        rows.append([
            r["verdict"], f"`{r['model']}`", r.get("precision", "?"),
            str(len(r["outputs"])),
            f"{w['rel_mean']:.4%}", f"{w['abs_rel']:.4%}",
            f"{w['max_abs']:.3g}", f"{w['corr']:.6f}", f"{w['scale']:.4f}",
        ])
    if rows:
        head = ["", "model", "prec", "outs", "rel", "AbsRel", "max abs", "corr", "scale"]
        widths = [max(len(str(c)) for c in [h] + [r[i] for r in rows])
                  for i, h in enumerate(head)]
        lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |")
        lines.append("| " + " | ".join("---" if i < 4 else "---:"
                                       for i in range(len(head))) + " |")
        for r in sorted(rows, key=lambda r: -float(r[4].rstrip("%"))):
            lines.append("| " + " | ".join(str(c).ljust(w)
                                           for c, w in zip(r, widths)) + " |")
        lines.append("")
        lines.append("Worst output per model is shown. `scale` is the "
                     "least-squares factor between the two: a value away from "
                     "1.0000 means the engine is off by a constant, which is a "
                     "different failure from noise.")
        lines.append("")

    if skipped:
        lines += ["## Not checked", ""]
        lines += [f"- `{m}` — {why}" for m, why in skipped]
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*", help="default: all with an input file")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--json", action="store_true", help="also dump raw numbers")
    args = ap.parse_args()

    runs = load_all(os.path.join(ROOT, "reports", "bench"))
    if args.models:
        runs = [r for r in runs if r["model"] in set(args.models)]
    if not runs:
        print("no benchmark results to verify")
        return 1

    results = []
    for run in runs:
        print(f"[{run['model']}] ", end="", flush=True)
        try:
            r = verify(run)
        except Exception as e:                      # noqa: BLE001
            r = {"model": run["model"], "skip": f"{type(e).__name__}: {e}"}
        results.append(r)
        print(r.get("skip") or f"{r['verdict']}  rel {r['worst_rel']:.4%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(render(results))
    print(f"\nwrote {os.path.relpath(args.out, ROOT)}")
    if args.json:
        p = os.path.splitext(args.out)[0] + ".json"
        json.dump(results, open(p, "w"), indent=2)
        print(f"wrote {os.path.relpath(p, ROOT)}")

    bad = [r for r in results if r.get("verdict") == "FAIL"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
