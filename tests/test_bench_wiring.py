"""Every onnx2trt.py must report through core/bench.py.

The point of Phase 3 is that no benchmark number is ever typed by hand again.
That only holds while all twelve scripts go through the shared loop, and the
easy way to lose it is to copy an old script as the template for a new model.
These checks are static -- no GPU, no engine -- so they run anywhere.
"""

import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


MODELS = os.path.join(ROOT, "models")


def scripts():
    for d in sorted(os.listdir(MODELS)):
        p = os.path.join(MODELS, d, "onnx2trt.py")
        if os.path.isfile(p):
            yield d, p, open(p, encoding="utf-8").read()


def benchmark_scripts():
    """The scripts whose job is to produce a comparable number.

    onnx2trt.py for every model, plus vggt/onnx2trt_split.py, which times a
    second way of running the same model and was overlooked: it kept a
    time.time() loop of its own for months, and the number it printed could not
    honestly be put beside the single-engine number.

    The onnx2trt_pointcloud.py and onnx2trt_video.py demos are deliberately not
    here. They print timings too, but at other resolutions and with
    post-processing inside the loop; wiring them to record() would put rows in
    reports/bench that mean something different from every other row.
    """
    for d, p, src in scripts():
        yield d, p, src
    extra = os.path.join(MODELS, "vggt", "onnx2trt_split.py")
    if os.path.isfile(extra):
        yield "vggt/split", extra, open(extra, encoding="utf-8").read()


def test_every_model_is_wired():
    n = 0
    for d, _, src in benchmark_scripts():
        n += 1
        check(f"{d} imports bench", "from core import bench" in src)
        check(f"{d} times through bench.measure", "bench.measure(" in src)
        check(f"{d} reports through bench.record", "bench.record(" in src)
    check("found the model scripts", n >= 13, f"only {n}")


def test_a_second_variant_does_not_overwrite_the_first():
    """Two ways of running one model share a record key, so the file name has
    to separate them -- bench.save() puts `variant` in it. Without that the
    split result would land on top of the single-engine one and the comparison
    would lose the thing it was comparing against."""
    src = open(os.path.join(MODELS, "vggt", "onnx2trt_split.py"),
               encoding="utf-8").read()
    check("split declares a variant", re.search(r"variant\s*=\s*'split'", src)
          is not None, "bench.record() without variant= overwrites vggt's row")


def test_no_hand_rolled_timing_remains():
    """A leftover manual loop would keep printing a number that never reaches
    reports/bench, so the table would silently omit that model."""
    for d, _, src in scripts():
        check(f"{d} has no dur_time", not re.search(r"\bdur_time\b", src))
        check(f"{d} has no avg_time", not re.search(r"\bavg_time\b", src))
        check(f"{d} does not call time.time in a loop",
              "time.time()" not in src)


def test_warmup_and_iterations_agree_across_models():
    """Different warmup counts alone make two models incomparable. They used
    to be 5, 10 and 20 depending on the file."""
    warms, iters = {}, {}
    for d, _, src in scripts():
        w = re.search(r"^\s*warmup\s*=\s*(\d+)", src, re.M)
        i = re.search(r"^\s*iteration\s*=\s*(\d+)", src, re.M)
        check(f"{d} declares warmup", bool(w))
        check(f"{d} declares iteration", bool(i))
        if w:
            warms.setdefault(int(w.group(1)), []).append(d)
        if i:
            iters.setdefault(int(i.group(1)), []).append(d)
    check("one warmup value across models", len(warms) == 1, str(warms))
    check("one iteration count across models", len(iters) == 1, str(iters))


def test_record_names_the_model():
    """The first argument to record() becomes the key in the report, so a
    copy-paste that leaves the donor's name silently overwrites its file."""
    for d, p, src in scripts():
        m = re.search(r"bench\.record\(\s*'([a-z0-9_]+)'", src)
        check(f"{d} record() has a literal model key", bool(m), src[:0])
        if not m:
            continue
        key, folder = m.group(1), d.lower()
        # unidepth_v2 lives in unidepth_v2; compare.py declares that
        from compare import FOLDER
        expected = {v.lower(): k for k, v in FOLDER.items()}.get(folder, folder)
        check(f"{d} record key matches the folder", key == expected,
              f"record says '{key}', expected '{expected}'")


def test_record_call_parses():
    """A malformed record() would only fail at runtime, on the GPU box, after
    the engine build -- the most expensive place to find a typo."""
    for d, p, src in scripts():
        tree = ast.parse(src)
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "record"]
        check(f"{d} has exactly one record() call", len(calls) == 1, str(len(calls)))
        if not calls:
            continue
        kw = {k.arg for k in calls[0].keywords}
        for required in ("warmup", "precision", "profile", "input_h", "input_w"):
            check(f"{d} record() passes {required}", required in kw, str(sorted(kw)))


def test_measure_wraps_do_inference_only():
    """If post-processing crept back inside the lambda, that model would be
    timing something the others do not. This is what depth_pro did."""
    for d, p, src in scripts():
        m = re.search(r"bench\.measure\((.*?)warmup=", src, re.S)
        check(f"{d} measure() body found", bool(m))
        if not m:
            continue
        body = m.group(1)
        check(f"{d} measure() wraps only do_inference",
              body.count("do_inference") == 1 and "interpolate" not in body,
              body.strip()[:160])


def test_onnx2trt_runs_in_the_shared_env():
    """Everything after the ONNX file is meant to run in one `trte` env.

    Three scripts used to break that by importing a model's upstream package
    at module scope without ever calling it -- metric3d_v2 imported UniDepth,
    of all things. A hard import means the script cannot even start unless
    that package is installed, so the shared environment stops being shared.

    moge_2 and metric_anything are the documented exceptions: they really do
    call MoGe's recover_focal_shift as post-process.
    """
    ALLOWED = {"moge_2", "metric_anything"}
    # upstream project roots, not pip packages like utils3d or trimesh
    UPSTREAM = {"unidepth", "UniDepth", "unik3d", "unik3d", "MoGe", "moge",
                "metric_anything", "depth_anything_v2", "vggt", "streamvggt",
                "Metric3D", "depth_pro", "distillanydepth"}

    for d, _, src in scripts():
        tree = ast.parse(src)
        need = set()
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.module:
                mods = [n.module]
            elif isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            need |= {m for m in mods if m.split(".")[0] in UPSTREAM}
        if d in ALLOWED:
            check(f"{d} is a known exception", bool(need),
                  "no upstream import left - drop it from ALLOWED")
        else:
            check(f"{d} needs no upstream package", not need, str(sorted(need)))


def test_no_unused_upstream_imports():
    """An upstream import that is never called is pure cost: it pins a package
    into the shared env for nothing. This is how all three of the above got in."""
    UPSTREAM = {"unidepth", "UniDepth", "unik3d", "unik3d", "MoGe", "moge",
                "metric_anything", "Metric3D", "depth_pro", "distillanydepth"}
    for d, _, src in scripts():
        tree = ast.parse(src)
        bound = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module \
                    and n.module.split(".")[0] in UPSTREAM:
                for a in n.names:
                    bound[a.asname or a.name] = n.module
        if not bound:
            continue
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        dead = sorted(k for k in bound if k not in used)
        check(f"{d} has no dead upstream import", not dead,
              f"{dead} from {sorted(set(bound.values()))}")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
