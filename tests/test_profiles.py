"""Keep onnx_export.py and onnx2trt.py agreeing on the input size.

The engine input shape is static and the resolution is part of the model name,
so a mismatch means onnx2trt.py looks for an ONNX that was never written. That
is exactly what happened to moge_2 (exported 291x518, loaded 388x518).

Also checks the models that carry a bench/native profile switch: both files
must offer the same profiles and the same size for each.
"""

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


def code_of(rel):
    """File contents with comments and docstrings stripped."""
    path = os.path.join(ROOT, rel.replace("/", os.sep))
    if not os.path.exists(path):
        return None
    src = open(path, encoding="utf-8").read()
    src = re.sub(r'"""[\s\S]*?"""', '""', src)
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))


def sizes_in(code):
    """Every (h, w) the file pins, from either spelling."""
    out = []
    for m in re.finditer(r'input_h\s*,\s*input_w\s*=\s*\(?\s*(\d+)\s*,\s*(\d+)', code):
        out.append((int(m.group(1)), int(m.group(2))))
    h = re.search(r'^\s*input_h\s*=\s*(\d+)', code, re.M)
    w = re.search(r'^\s*input_w\s*=\s*(\d+)', code, re.M)
    if h and w:
        out.append((int(h.group(1)), int(w.group(1))))
    return out


# models whose two scripts must pin the same size
PAIRS = [
    "Depth_Anything_V2", "Depth_Anything_AC", "Depth_Anything_V3",
    "Distill_Any_Depth", "Metric3D_V2", "MoGe_2", "StreamVGGT",
    "Uni_Depth_V2", "UniK3D", "VGGT",
]

# models offering both profiles
PROFILED = ["Depth_Anything_AC", "Uni_Depth_V2", "UniK3D"]


def test_export_and_trt_agree():
    for model in PAIRS:
        exp, trt = code_of(f"{model}/onnx_export.py"), code_of(f"{model}/onnx2trt.py")
        if exp is None or trt is None:
            check(f"{model} both scripts exist", False, "missing file")
            continue
        se, st = set(sizes_in(exp)), set(sizes_in(trt))
        if not se or not st:
            check(f"{model} sizes found", False, f"export={se} trt={st}")
            continue
        check(f"{model} sizes agree", se == st, f"export={sorted(se)} trt={sorted(st)}")


def test_profiled_models_offer_both():
    for model in PROFILED:
        for script in ("onnx_export.py", "onnx2trt.py"):
            code = code_of(f"{model}/{script}")
            has_switch = bool(re.search(r"profile\s*=\s*'(bench|native)'", code))
            has_both = "'bench'" in code and "'native'" in code
            check(f"{model}/{script} has a profile switch", has_switch and has_both)


def test_bench_is_518_square():
    """The comparison profile has to be identical across models."""
    for model in PROFILED:
        code = code_of(f"{model}/onnx2trt.py")
        m = re.search(r"\(518,\s*518\)\s*if\s*profile\s*==\s*'bench'", code)
        check(f"{model} bench is 518x518", bool(m))


def test_moge_regression():
    """The mismatch that motivated this file."""
    exp, trt = code_of("MoGe_2/onnx_export.py"), code_of("MoGe_2/onnx2trt.py")
    check("moge_2 export matches trt", set(sizes_in(exp)) == set(sizes_in(trt)),
          f"export={sorted(set(sizes_in(exp)))} trt={sorted(set(sizes_in(trt)))}")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
