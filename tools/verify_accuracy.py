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
    python verify_accuracy.py                 # every model -> reports/accuracy.md
    python verify_accuracy.py depth_anything_v2 unik3d
                                              # -> reports/accuracy_<models>.md
    python verify_accuracy.py --precision fp32

Naming a subset writes its own file. Only a full run replaces the full table,
so checking one engine cannot delete the other eleven rows.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import unmeasured  # noqa: E402
from core.bench import load_all  # noqa: E402
from core.golden import best_scale, compare  # noqa: E402
from compare import FOLDER  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def extra_inputs_for(model):
    """{name: array} for a graph that takes more than an image.

    Written by core.bench.record as reports/inputs/<model>__<name>.npy. Only
    tr2m has any today -- its [1,1,768] CLIP text embedding -- and the point of
    keeping them beside the image tensor is that the reference run feeds the
    engine exactly what the benchmark fed it, second input included. Feeding a
    zero vector instead would compare two different computations and report the
    difference as fp16 error.
    """
    found = {}
    for path in sorted(glob.glob(os.path.join(INPUTS, f"{model}__*.npy"))):
        name = os.path.basename(path)[len(model) + 2:-4]
        found[name] = np.load(path)
    return found


def run_onnx(path, x, extras=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    # CPU deliberately: the point is a trustworthy fp32 reference, not speed,
    # and the CUDA provider may pick different kernels than the CPU one.
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    feed = {names[0]: x.astype(np.float32)}
    for name in names[1:]:
        if name not in (extras or {}):
            raise ValueError(
                f"{path} wants an input named {name!r} and nothing saved it. "
                f"Pass it to bench.record(extra_inputs=...) in the model's "
                f"onnx2trt.py so the reference sees what the engine saw.")
        feed[name] = np.asarray(extras[name], dtype=np.float32)
    outs = sess.run(None, feed)
    return [np.asarray(o) for o in outs]


def run_trt(engine_path, x, out_shapes, extras=None):
    """Deserialize the built engine and run it once.

    Deliberately not common.get_engine: that checks a fingerprint and rebuilds
    when it does not match, which for a verification pass would mean silently
    measuring a freshly built engine instead of the one that produced the
    benchmark numbers.
    """
    import tensorrt as trt
    from core import common

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"could not deserialize {engine_path}")

    with engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = np.ascontiguousarray(x, dtype=np.float32)
        # Extra inputs follow the engine's binding order, which is the ONNX
        # input order, which is the order they were saved in.
        for slot, value in zip(inputs[1:], (extras or {}).values()):
            slot.host = np.ascontiguousarray(value, dtype=np.float32)
        raw = common.do_inference(context, engine=engine, bindings=bindings,
                                  inputs=inputs, outputs=outputs, stream=stream)
        got = [np.array(r[: int(np.prod(s))]).reshape(s)
               for r, s in zip(raw, out_shapes)]
        common.free_buffers(inputs, outputs, stream)
        return got


def verify(run, onnx_override=None):
    model = run["model"]
    xp = os.path.join(INPUTS, f"{model}.npy")
    if not os.path.exists(xp):
        return {"model": model, "skip": "no reports/inputs/*.npy — rerun onnx2trt.py"}
    onnx_file = onnx_override or onnx_path_for(run)
    if not onnx_file:
        return {"model": model, "skip": f"onnx not found next to {run.get('engine_path')}"}
    engine = run.get("engine_path")
    if not os.path.exists(engine):
        return {"model": model, "skip": f"engine missing: {engine}"}

    x = np.load(xp)
    extras = extra_inputs_for(model)
    ref = run_onnx(onnx_file, x, extras)
    got = run_trt(engine, x, [o.shape for o in ref], extras)

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
            "ref_mag": float(np.abs(finite).mean()) if m.any() else float("nan"),
            "rel_mean": rep["rel_mean"],
            "abs_rel": rep["abs_rel"],
            "rmse": rep["rmse"],
            "max_abs": float(np.abs(a[m] - b[m]).max()) if m.any() else float("nan"),
            "corr": corr,
            "scale": 1.0 / best_scale(a, b, m) if m.any() else float("nan"),
        })
    for o in per_output:
        o["verdict"] = verdict_for(o)
    rank = {"FAIL": 2, "WARN": 1, "ok": 0}
    worst = max(per_output, key=lambda o: (rank[o["verdict"]], o["rel_mean"]))
    return {"model": model, "precision": run.get("precision"),
            "onnx": os.path.relpath(onnx_file, ROOT),
            "outputs": per_output,
            "worst_rel": max(o["rel_mean"] for o in per_output),
            "verdict": worst["verdict"]}


def verdict_for(o):
    """Judge an output on relative error, but not on that alone.

    `rel` divides by the mean magnitude of the reference, so an output whose
    values sit near zero inflates it for free. Depth Anything V3's sky mask
    scores 7.84% while differing by at most 0.012 in absolute terms — less than
    the depth output's 0.032, which scores 0.11%. Calling the first a failure
    and the second a pass gets it backwards.

    So a large `rel` is a failure only when the two also stop agreeing in
    shape. High correlation with a tiny absolute difference is fp16 noise on a
    small-valued map, not a broken engine.
    """
    if not np.isfinite(o["rel_mean"]):
        return "ok"
    structural = np.isfinite(o["corr"]) and o["corr"] < 0.99
    if o["rel_mean"] > FAIL:
        return "FAIL" if structural else "WARN"
    if o["rel_mean"] > WARN:
        return "WARN" if structural else "ok"
    return "ok"


def input_size(model):
    """(h, w) of the tensor this model's engine was compared on.

    Straight off reports/inputs/<model>.npy, which is the array verify_accuracy
    actually feeds both the engine and the ONNX graph -- not a number parsed out
    of a filename and not one retyped from spec.json. Memory-mapped, so this
    reads a header rather than 28 MB.

    Execution rule 8: the table ranked twelve models by `rel` with no column
    saying that depth_pro was measured on 1536x1536 and moge_2 on 388x518.
    """
    path = os.path.join(INPUTS, f"{model}.npy")
    if not os.path.exists(path):
        return None
    shape = np.load(path, mmap_mode="r").shape
    return (shape[-2], shape[-1]) if len(shape) >= 2 else None


def render(results):
    # An engine nobody has verified is a record here, not an absence. Twelve of
    # the fourteen models have a result in reports/accuracy.json and the other
    # two were simply not in the file, so the table's row count was the only
    # thing that said so -- and PLAN.md read it as fourteen. core/unmeasured.py
    # explains the record; the point of splitting here is that `missing` has to
    # be rendered somewhere below or the reader is back where they started.
    results, missing = unmeasured.split(results)
    lines = ["# Engine accuracy", "",
             "Each TensorRT engine against the ONNX graph it was built from, "
             "at fp32, on the byte-identical input the benchmark used "
             "(`reports/inputs/`).", "",
             "This isolates what TensorRT and fp16 did. It is **not** a "
             "comparison against PyTorch — that would fold in preprocessing "
             "and the ONNX exporter as well. Generated by `verify_accuracy.py`.",
             ""]

    rows, skipped = [], []
    for r in results:
        if "skip" in r:
            skipped.append((r["model"], r["skip"]))
            continue
        for o in r["outputs"]:
            # A JSON from an earlier run predates these fields. Filling them in
            # here means --from-json can re-render an old result rather than
            # forcing every engine to run again just to reword a verdict.
            o.setdefault("verdict", verdict_for(o))
            o.setdefault("ref_mag", float("nan"))
            hw = input_size(r["model"])
            rows.append([
                o["verdict"], f"`{r['model']}`",
                "-" if hw is None else f"{hw[0]}x{hw[1]}",
                r.get("precision", "?"),
                str(o["index"]),
                f"{o['ref_mag']:.4g}" if o["ref_mag"] == o["ref_mag"] else "-",
                f"{o['rel_mean']:.4%}", f"{o['max_abs']:.3g}",
                f"{o['corr']:.6f}" if o["corr"] == o["corr"] else "-",
                f"{o['scale']:.4f}",
            ])
    if rows:
        # `ref \|mean\|`, escaped. Unescaped, the two pipes inside that
        # header cell split it into three, so the header had twelve columns
        # against the rows' ten and every markdown renderer misaligned the
        # whole table -- the published file has been wrong since the column was
        # added. The escape is what the cell always meant to say.
        head = ["", "model", "input HxW", "prec", "out", r"ref \|mean\|", "rel",
                "max abs", "corr", "scale"]
        widths = [max(len(str(c)) for c in [h] + [r[i] for r in rows])
                  for i, h in enumerate(head)]
        lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |")
        lines.append("| " + " | ".join("---" if i < 5 else "---:"
                                       for i in range(len(head))) + " |")
        # index 6 is `rel`, which moved when input HxW was inserted
        for r in sorted(rows, key=lambda r: -float(r[6].rstrip("%"))):
            lines.append("| " + " | ".join(str(c).ljust(w)
                                           for c, w in zip(r, widths)) + " |")
        lines += ["", "One row per model output, worst first.", "",
                  "**`input HxW` is a condition, not a result.** These rows are "
                  "not a ranking of models: depth_pro is compared on 1536x1536 "
                  "and moge_2 on 388x518, and `rel` is a property of one engine "
                  "against its own graph at that size. The size is read from "
                  "the shape of `reports/inputs/<model>.npy`, the array both "
                  "sides were fed; every other number here comes out of "
                  "`reports/accuracy.json`.", ""]

    lines += [
        "**Reading `rel`.** It is the mean absolute difference divided by the "
        "mean magnitude of the reference, so an output whose values sit near "
        f"zero inflates it for free. Flagged above {WARN:.0%} and failed above "
        f"{FAIL:.0%}, but only when the correlation also drops below 0.99 — "
        "otherwise a large `rel` on a small-valued map is fp16 noise, not a "
        "broken engine. `ref |mean|` is there so that judgement can be checked.",
        "",
        "`scale` is the least-squares factor between the two. A value away "
        "from 1.0000 means the engine is off by a constant, which is a "
        "different failure from noise.",
        "",
    ]

    if skipped:
        lines += ["## Not checked", ""]
        lines += [f"- `{m}` — {why}" for m, why in skipped]
        lines.append("")
    # Separate from "Not checked" on purpose. That section is about a run that
    # happened and could not finish a model; this one is about a run that was
    # never started, which is a different thing to do something about.
    lines += unmeasured.section(
        missing,
        intro="No engine-against-ONNX check exists for these. They are listed "
              "because the table above has one row per *output*, so a model "
              "with no result does not leave a visible gap in it.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*", help="default: all with an input file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true", help="also dump raw numbers")
    ap.add_argument("--from-json", metavar="PATH",
                    help="re-render the report from a previous run, no GPU needed")
    ap.add_argument("--onnx", default=None,
                    help="compare this ONNX instead of the recorded graph")
    ap.add_argument("--engine", default=None,
                    help="compare this engine instead of the recorded engine")
    args = ap.parse_args()

    if bool(args.onnx) != bool(args.engine):
        ap.error("--onnx and --engine must be supplied together")
    if args.onnx and len(args.models) != 1:
        ap.error("a custom ONNX/engine pair requires exactly one model")

    # Checking one model used to overwrite reports/accuracy.md with a table
    # holding only that model, silently deleting the other eleven rows -- which
    # happened twice while chasing unik3d. A subset now lands in its own file
    # unless the caller asks for otherwise, so the full table is only ever
    # replaced by a full run.
    if args.out is None:
        args.out = OUT if not args.models else os.path.join(
            os.path.dirname(OUT),
            "accuracy_" + "_".join(sorted(args.models)) + ".md")
    if args.from_json:
        # Re-rendering costs nothing and needs no GPU, so a change to how
        # the verdict is worded does not mean re-running every engine.
        results = json.load(open(args.from_json, encoding='utf-8'))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, 'w', encoding='utf-8').write(render(results))
        print(f'wrote {os.path.relpath(args.out, ROOT)} from {args.from_json}')
        return 0

    runs = load_all(os.path.join(ROOT, "reports", "bench"))
    if args.models:
        runs = [r for r in runs if r["model"] in set(args.models)]
    if args.engine and runs:
        runs[0] = dict(runs[0], engine_path=os.path.abspath(args.engine))
    if not runs:
        print("no benchmark results to verify")
        return 1

    results = []
    for run in runs:
        print(f"[{run['model']}] ", end="", flush=True)
        try:
            r = verify(run, os.path.abspath(args.onnx) if args.onnx else None)
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
