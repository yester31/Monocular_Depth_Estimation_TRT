"""Tests for core/golden.py — the reference capture used to prove that fixing
a defect changed output the way it was meant to."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.golden import compare, format_report, load_reference, save_reference  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


def test_roundtrip():
    rng = np.random.default_rng(0)
    outs = {"depth": rng.random((4, 8), np.float32),
            "conf": rng.random((4, 8), np.float32),
            "scale": np.float32([2.5])}
    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "in.png")
        open(img, "wb").write(b"fake image bytes")
        p = save_reference(td, "demo_model", outs, input_path=img,
                           input_shape=(480, 640, 3), profile="bench")
        check("npz written", os.path.exists(p))
        check("json written", os.path.exists(os.path.splitext(p)[0] + ".json"))

        back, meta = load_reference(p)
        check("keys preserved", set(back) == set(outs))
        check("values preserved", all(np.array_equal(back[k], outs[k]) for k in outs))
        check("input hashed", bool(meta.get("input_sha256")))
        check("stats recorded", meta["outputs"]["depth"]["shape"] == [4, 8])
        check("filename encodes config",
              os.path.basename(p) == "demo_model_single_bench_torch_fp32.npz",
              os.path.basename(p))


def test_compare_identical():
    a = {"depth": np.arange(12, dtype=np.float32).reshape(3, 4)}
    rep = compare(a, {k: v.copy() for k, v in a.items()})
    check("identical detected", rep["depth"]["identical"] is True)
    check("max_abs zero", rep["depth"]["max_abs"] == 0.0)


def test_compare_changed():
    a = {"depth": np.full((3, 4), 2.0, np.float32)}
    b = {"depth": np.full((3, 4), 2.1, np.float32)}
    rep = compare(a, b)
    r = rep["depth"]
    check("difference measured", abs(r["max_abs"] - 0.1) < 1e-6, str(r["max_abs"]))
    check("relative measured", abs(r["rel_mean"] - 0.05) < 1e-3, str(r["rel_mean"]))
    check("not identical", r["identical"] is False)


def test_compare_handles_inf_and_nan():
    """moge_2 and metric_anything mark invalid pixels with inf; one such pixel
    must not swamp the statistics."""
    a = np.full((4, 4), 1.0, np.float32)
    b = a.copy()
    a[0, 0] = np.inf
    b[0, 0] = np.inf
    b[1, 1] = np.nan
    rep = compare({"pts": a}, {"pts": b})
    r = rep["pts"]
    check("inf/nan excluded", r["excluded"] == 2, str(r["excluded"]))
    check("rest compared", r["compared"] == 14, str(r["compared"]))
    check("finite part identical", r["max_abs"] == 0.0)


def test_compare_shape_mismatch():
    rep = compare({"d": np.zeros((2, 2))}, {"d": np.zeros((3, 3))})
    check("shape mismatch flagged", rep["d"]["status"] == "shape_mismatch")


def test_compare_missing_key():
    rep = compare({"a": np.zeros(2), "b": np.zeros(2)}, {"a": np.zeros(2), "c": np.zeros(2)})
    check("missing output flagged", rep["b"]["status"] == "missing")
    check("new output flagged", rep["c"]["status"] == "only_in_new")


def test_valid_mask():
    a = np.zeros((4, 4), np.float32)
    b = a.copy()
    b[2:, :] = 99.0                       # differs only where the mask is False
    mask = np.zeros((4, 4), bool)
    mask[:2, :] = True
    rep = compare({"d": a}, {"d": b}, valid_mask=mask)
    check("masked region ignored", rep["d"]["max_abs"] == 0.0, str(rep["d"]["max_abs"]))
    check("mask limits count", rep["d"]["compared"] == 8, str(rep["d"]["compared"]))


def test_format_report():
    rep = compare({"d": np.zeros((2, 2))}, {"d": np.ones((2, 2))})
    txt = format_report(rep)
    check("report mentions output", "d" in txt and "max" in txt, txt)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
