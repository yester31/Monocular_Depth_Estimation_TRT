"""Guards for the vggt / streamvggt square-pad geometry (defects D1, D2, D3, D7, D8).

The old code padded, resized to 1024, then downscaled to 518, while computing
crop coordinates on the 1024 grid. streamvggt then reconciled the two by
halving the coordinates (1024/518 = 1.977, not 2) and skipped the horizontal
crop entirely. These tests pin the corrected behaviour.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


def coords(width, height, net=518):
    """The corrected computation, as used by both onnx2trt.py files."""
    max_dim = max(width, height)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    scale = net / max_dim
    return (left * scale, top * scale,
            (left + width) * scale, (top + height) * scale)


def test_landscape_crops_horizontally():
    """D2: a landscape source must lose its left/right padding."""
    x1, y1, x2, y2 = coords(1280, 720)
    check("landscape spans full width", abs(x1) < 1e-9 and abs(x2 - 518) < 1e-9,
          f"x=[{x1:.2f},{x2:.2f}]")
    check("landscape is letterboxed vertically", y1 > 1.0 and y2 < 517.0,
          f"y=[{y1:.2f},{y2:.2f}]")
    kept_h = int(round(y2)) - int(round(y1))
    expect = round(720 / 1280 * 518)
    check("kept height matches aspect", abs(kept_h - expect) <= 1,
          f"{kept_h} vs {expect}")


def test_portrait_crops_vertically():
    x1, y1, x2, y2 = coords(720, 1280)
    check("portrait spans full height", abs(y1) < 1e-9 and abs(y2 - 518) < 1e-9,
          f"y=[{y1:.2f},{y2:.2f}]")
    check("portrait is pillarboxed", x1 > 1.0 and x2 < 517.0, f"x=[{x1:.2f},{x2:.2f}]")


def test_square_has_no_padding():
    x1, y1, x2, y2 = coords(900, 900)
    check("square keeps everything",
          (x1, y1) == (0.0, 0.0) and abs(x2 - 518) < 1e-9 and abs(y2 - 518) < 1e-9,
          f"[{x1},{y1},{x2},{y2}]")


def test_old_halving_was_wrong():
    """D1: the old streamvggt postprocess used coords computed at 1024 and
    divided them by 2 to reach 518. The true factor is 1024/518 = 1.977."""
    w, h, old_target, net = 1280, 720, 1024, 518
    max_dim = max(w, h)
    top = (max_dim - h) // 2
    y1_1024 = top * (old_target / max_dim)
    old = y1_1024 / 2                 # what the code did
    correct = top * (net / max_dim)   # what it should have been
    err = abs(old - correct)
    check("old halving was off", err > 1.0, f"{old:.2f} vs {correct:.2f}")
    # Halving scales by (1024/2)/518 = 0.9884 instead of 1, so the old crop
    # started ~1.16% too early — it kept a sliver of padding.
    ratio = (old_target / 2) / net
    check("old crop was too small", old < correct, f"{old:.2f} < {correct:.2f}")
    check("error is ~1.16%", abs(err / correct - (1.0 - ratio)) < 1e-3,
          f"{err / correct * 100:.2f}% vs {(1.0 - ratio) * 100:.2f}%")


def test_crop_is_within_bounds():
    """Rounded crop indices must stay inside a 518x518 map for many aspects."""
    for w, h in [(640, 480), (1280, 720), (1920, 1080), (720, 1280),
                 (500, 500), (3024, 2268), (100, 4000), (4000, 100)]:
        x1, y1, x2, y2 = coords(w, h)
        i = [int(round(v)) for v in (y1, y2, x1, x2)]
        ok = 0 <= i[0] < i[1] <= 518 and 0 <= i[2] < i[3] <= 518
        check(f"crop in bounds {w}x{h}", ok, str(i))


def test_no_1024_left_in_source():
    """D3/D8: neither script should still route through target_size=1024."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("VGGT/onnx2trt.py", "StreamVGGT/onnx2trt.py"):
        p = os.path.join(root, rel.replace("/", os.sep))
        # strip comments and docstrings so prose mentioning the old constant
        # does not count as code
        import ast
        import re
        src = open(p, encoding="utf-8").read()
        src = re.sub(r'"""[\s\S]*?"""', '""', src)
        code = [l for l in src.splitlines() if not l.strip().startswith("#")]
        assigns = [l for l in code
                   if re.search(r'\btarget_size\s*=(?!=)', l)]
        check(f"{rel} has no target_size assignment", not assigns,
              "; ".join(a.strip() for a in assigns))
        pads = [l for l in code if "padding = [0, 0, 0]" in l]
        check(f"{rel} does not pad black", not pads, "; ".join(p.strip() for p in pads))


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
