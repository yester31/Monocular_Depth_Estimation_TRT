"""How much does pre-resizing the input cost this model?

A static engine needs one input size, so every image is resized to it before
the model sees it. For some models that is free: relative depth barely notices.
For others it is not, and the reason is not obvious from the code -- unik3d
reads a 518x518 image as a picture actually taken at that size, infers a focal
length five times too short, and returns depth 3.15x off. That was measured,
not predicted.

docs/model_contracts.md 5.7 has the table. Six models are in it. This module
is the method it used, written down so the remaining ones are measured the same
way rather than a slightly different way:

  1. run the model on the original image, letting its own rule pick the size
  2. run it again on an image pre-resized to the engine's size
  3. bring both depth maps back to the original resolution
  4. compare, and separately report the scale factor between them

Step 4 is why scale is reported apart from the error. A depth map that is
right in every proportion but twice as large is a different failure from one
that is the right size and the wrong shape, and one number cannot say which
happened: rel_mean would be about 100% either way.

Nothing here imports torch or a model package -- it takes depth maps.
"""

import numpy as np

from core import golden


def to_original(depth, shape):
    """Resample a depth map back to the original resolution.

    cv2 is used when it is there and numpy indexing otherwise, because this
    runs inside twelve different environments and not all of them have it.
    Nearest-neighbour on the fallback path is cruder than the bilinear the
    models use, but it is only used to compare two maps against each other,
    and both go through the same path.
    """
    h, w = shape[:2]
    a = np.asarray(depth, dtype=np.float64)
    if a.shape[:2] == (h, w):
        return a
    try:
        import cv2
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        yi = (np.arange(h) * a.shape[0] // h).clip(0, a.shape[0] - 1)
        xi = (np.arange(w) * a.shape[1] // w).clip(0, a.shape[1] - 1)
        return a[yi][:, xi]


def measure(reference, candidates, original_shape):
    """reference: depth from the model's own rule. candidates: {label: depth}.

    Returns one row per candidate, each with the scale factor, the error as
    given, and the error once the scale is divided out. A model whose only
    problem is scale shows a large `rel_mean` and a small `rel_mean_scaled`;
    one whose structure moved shows both.
    """
    ref = to_original(reference, original_shape)
    rows = []
    for label, depth in candidates.items():
        got = to_original(depth, original_shape)
        # best_scale returns the s that makes s*got match ref, so the factor
        # the candidate is off by is its reciprocal. 5.7 reports it that way --
        # unik3d at 518x518 reads as 3.15x too large, not 0.317x.
        s = golden.best_scale(ref, got)
        raw = golden.compare({"depth": ref}, {"depth": got})["depth"]
        scaled = golden.compare({"depth": ref}, {"depth": got * s})["depth"]
        rows.append({
            "input": label,
            "scale": round(1.0 / s if s else float("inf"), 4),
            "rel_mean": round(raw.get("rel_mean", float("nan")), 6),
            "rel_mean_scaled": round(scaled.get("rel_mean", float("nan")), 6),
            "abs_rel": round(raw.get("abs_rel", float("nan")), 6),
            "delta1": round(raw.get("delta1", float("nan")), 6),
            "corr": round(float(np.corrcoef(ref.ravel(), got.ravel())[0, 1]), 6),
        })
    return rows


def render(model, rows):
    out = [f"{model}",
           f"  {'input':<14}{'scale':>8}{'rel':>10}{'rel scaled':>12}"
           f"{'abs_rel':>10}{'delta1':>9}{'corr':>9}"]
    for r in rows:
        out.append(f"  {r['input']:<14}{r['scale']:>8.3f}"
                   f"{r['rel_mean'] * 100:>9.2f}%{r['rel_mean_scaled'] * 100:>11.2f}%"
                   f"{r['abs_rel'] * 100:>9.2f}%{r['delta1']:>9.4f}{r['corr']:>9.4f}")
    return "\n".join(out)
