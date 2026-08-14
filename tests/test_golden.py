"""Tests for core/golden.py — the reference capture used to prove that fixing
a defect changed output the way it was meant to."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.golden import (  # noqa: E402
    best_scale, compare, format_report, load_reference, save_reference,
)

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


def test_depth_metrics():
    """AbsRel / RMSE / delta1 alongside the aggregate ratio."""
    a = np.full((10, 10), 2.0)
    b = np.full((10, 10), 2.2)            # every pixel 10% high
    r = compare({"d": a}, {"d": b})["d"]
    check("abs_rel is per-pixel", abs(r["abs_rel"] - 0.1) < 1e-9, str(r["abs_rel"]))
    check("rmse", abs(r["rmse"] - 0.2) < 1e-9, str(r["rmse"]))
    check("delta1 all within 1.25", r["delta1"] == 1.0, str(r["delta1"]))

    c = np.full((10, 10), 6.0)            # 3x -> outside 1.25
    r2 = compare({"d": a}, {"d": c})["d"]
    check("delta1 catches 3x", r2["delta1"] == 0.0, str(r2["delta1"]))
    check("rel_mean exceeds 100%", r2["rel_mean"] > 1.0, str(r2["rel_mean"]))


def test_depth_metrics_skip_nonpositive():
    """Dividing by a zero depth is meaningless, so those pixels are dropped."""
    a = np.array([[1.0, 0.0], [2.0, -1.0]])
    b = np.array([[1.1, 0.0], [2.2, -1.0]])
    r = compare({"d": a}, {"d": b})["d"]
    check("only positive pixels used", r["positive"] == 2, str(r.get("positive")))
    check("abs_rel unaffected by zeros", abs(r["abs_rel"] - 0.1) < 1e-9, str(r["abs_rel"]))


def test_best_scale():
    """A pure scale change should be recoverable exactly."""
    rng = np.random.default_rng(1)
    a = rng.random((50, 50)) + 0.5
    b = a * 3.15
    s = best_scale(a, b)
    check("scale recovered", abs(s - 1 / 3.15) < 1e-9, str(s))
    resid = np.abs(a - s * b).mean() / np.abs(a).mean()
    check("residual ~0 after scaling", resid < 1e-12, str(resid))

    # a change that is NOT pure scale must leave a residual
    c = a * 3.15 + rng.random((50, 50)) * 0.5
    s2 = best_scale(a, c)
    resid2 = np.abs(a - s2 * c).mean() / np.abs(a).mean()
    check("structural change survives scaling", resid2 > 0.01, str(resid2))


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
