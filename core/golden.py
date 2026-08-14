"""Record and compare golden references.

Fixing the defects in docs/model_contracts.md changes model output — that is
the point. Without a snapshot of the current behaviour there is no way to tell
"changed as intended" from "broke something", so every model gets a reference
captured *before* its fix lands.

A reference is one .npz of raw output arrays plus a .json of metadata. Both are
written by the model's own conda environment; comparison happens later, in
whichever environment is convenient, since only numpy is needed to read them.
"""

import hashlib
import json
import os
import platform
import time
from typing import Dict, Optional

import numpy as np


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def save_reference(
    out_dir: str,
    model: str,
    outputs: Dict[str, np.ndarray],
    *,
    profile: str = "bench",
    variant: str = "single",
    backend: str = "torch",
    precision: str = "fp32",
    input_path: Optional[str] = None,
    input_shape=None,
    extra: Optional[dict] = None,
) -> str:
    """Write <out_dir>/<model>_<variant>_<profile>_<backend>_<precision>.{npz,json}."""
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{model}_{variant}_{profile}_{backend}_{precision}"
    npz_path = os.path.join(out_dir, stem + ".npz")

    arrays = {k: np.asarray(v) for k, v in outputs.items()}
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "model": model,
        "profile": profile,
        "variant": variant,
        "backend": backend,
        "precision": precision,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "input_path": input_path,
        "input_sha256": _sha256(input_path) if input_path and os.path.exists(input_path) else None,
        "input_shape": list(input_shape) if input_shape is not None else None,
        "outputs": {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                # summary stats, so an eyeball check does not need the arrays
                "min": float(np.nanmin(v)) if v.size else None,
                "max": float(np.nanmax(v)) if v.size else None,
                "mean": float(np.nanmean(v)) if v.size else None,
                "nan": int(np.isnan(v).sum()) if np.issubdtype(v.dtype, np.floating) else 0,
                "inf": int(np.isinf(v).sum()) if np.issubdtype(v.dtype, np.floating) else 0,
            }
            for k, v in arrays.items()
        },
    }
    if extra:
        meta["extra"] = extra
    try:
        import torch

        meta["torch"] = torch.__version__
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return npz_path


def load_reference(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta_path = os.path.splitext(path)[0] + ".json"
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    return arrays, meta


def compare(ref: Dict[str, np.ndarray], got: Dict[str, np.ndarray],
            valid_mask: Optional[np.ndarray] = None, depth_metrics: bool = True) -> dict:
    """Per-output difference metrics. Inputs must already be the same shape.

    NaN/Inf are excluded pairwise rather than propagated — several models emit
    Inf for invalid pixels by design, and a single Inf would otherwise swamp
    every statistic.

    Two families of number come out of this:

    `rel_mean` is mean|a-b| / mean|a| — one ratio for the whole map. It is
    cheap and never divides by a small value, but it cannot tell a constant
    scale factor from a broken structure, and it exceeds 100% as soon as the
    values roughly double.

    `abs_rel`, `rmse` and `delta1` are the standard depth-estimation metrics,
    computed per pixel, so they are comparable with what papers report.
    `delta1` in particular says something `rel_mean` cannot: the fraction of
    pixels within a factor of 1.25. Pixels at or below `eps` are dropped from
    these, since dividing by them is meaningless.
    """
    report = {}
    for name in sorted(set(ref) | set(got)):
        if name not in ref:
            report[name] = {"status": "only_in_new"}
            continue
        if name not in got:
            report[name] = {"status": "missing"}
            continue

        a, b = np.asarray(ref[name], np.float64), np.asarray(got[name], np.float64)
        if a.shape != b.shape:
            report[name] = {"status": "shape_mismatch",
                            "ref_shape": list(a.shape), "new_shape": list(b.shape)}
            continue

        finite = np.isfinite(a) & np.isfinite(b)
        if valid_mask is not None and valid_mask.shape == a.shape:
            finite &= valid_mask.astype(bool)
        n = int(finite.sum())
        if n == 0:
            report[name] = {"status": "no_finite_overlap"}
            continue

        av, bv = a[finite], b[finite]
        d = np.abs(av - bv)
        scale = np.abs(av).mean()
        entry = {
            "status": "ok",
            "compared": n,
            "excluded": int(a.size - n),
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "rel_mean": float(d.mean() / scale) if scale > 0 else None,
            "identical": bool(d.max() == 0),
        }

        if depth_metrics:
            eps = 1e-6
            pos = (av > eps) & (bv > eps)
            k = int(pos.sum())
            if k:
                ap, bp = av[pos], bv[pos]
                rel = np.abs(ap - bp) / ap
                ratio = np.maximum(ap / bp, bp / ap)
                entry.update({
                    "abs_rel": float(rel.mean()),
                    "abs_rel_median": float(np.median(rel)),
                    "rmse": float(np.sqrt(((ap - bp) ** 2).mean())),
                    "delta1": float((ratio < 1.25).mean()),
                    "positive": k,
                })
        report[name] = entry
    return report


def best_scale(ref: np.ndarray, got: np.ndarray,
               valid_mask: Optional[np.ndarray] = None) -> float:
    """Least-squares factor s minimising |ref - s*got|.

    Metric models infer focal length from the input, so feeding a different
    resolution shifts every depth by roughly a constant. Dividing that out
    separates "the scale moved" from "the geometry changed" — on unik3d at
    518x518 it turns a 215% error into 5.6%.
    """
    a, b = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if valid_mask is not None and valid_mask.shape == a.shape:
        m &= valid_mask.astype(bool)
    denom = float((b[m] * b[m]).sum())
    return float((a[m] * b[m]).sum() / denom) if denom > 0 else 1.0


def format_report(report: dict, indent="  ") -> str:
    lines = []
    for name, r in report.items():
        if r["status"] != "ok":
            lines.append(f"{indent}{name:<22} {r['status'].upper()} "
                         + (f"{r.get('ref_shape')} vs {r.get('new_shape')}"
                            if r["status"] == "shape_mismatch" else ""))
            continue
        rel = "" if r["rel_mean"] is None else f"  rel {r['rel_mean'] * 100:.4f}%"
        excl = f"  (excluded {r['excluded']})" if r["excluded"] else ""
        lines.append(f"{indent}{name:<22} max {r['max_abs']:.4e}  "
                     f"mean {r['mean_abs']:.4e}{rel}{excl}")
        if "abs_rel" in r:
            lines.append(f"{indent}{'':22} AbsRel {r['abs_rel'] * 100:.2f}%  "
                         f"RMSE {r['rmse']:.4f}  d1 {r['delta1'] * 100:.1f}%")
    return "\n".join(lines)
