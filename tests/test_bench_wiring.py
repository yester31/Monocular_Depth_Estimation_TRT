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


def scripts():
    for d in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, d, "onnx2trt.py")
        if os.path.isfile(p):
            yield d, p, open(p, encoding="utf-8").read()


def test_every_model_is_wired():
    n = 0
    for d, _, src in scripts():
        n += 1
        check(f"{d} imports bench", "from core import bench" in src)
        check(f"{d} times through bench.measure", "bench.measure(" in src)
        check(f"{d} reports through bench.record", "bench.record(" in src)
    check("found the model scripts", n >= 12, f"only {n}")


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
        # unidepth_v2 lives in Uni_Depth_V2; compare.py declares that
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


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
