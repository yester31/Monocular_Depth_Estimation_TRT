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
            valid_mask: Optional[np.ndarray] = None) -> dict:
    """Per-output difference metrics.

    NaN/Inf are excluded pairwise rather than propagated — several models emit
    Inf for invalid pixels by design, and a single Inf would otherwise swamp
    every statistic.
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

        d = np.abs(a[finite] - b[finite])
        scale = np.abs(a[finite]).mean()
        report[name] = {
            "status": "ok",
            "compared": n,
            "excluded": int(a.size - n),
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "rel_mean": float(d.mean() / scale) if scale > 0 else None,
            "identical": bool(d.max() == 0),
        }
    return report


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
    return "\n".join(lines)
