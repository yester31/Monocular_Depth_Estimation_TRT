"""Comparing a predicted depth map against measured ground truth.

Two things live here, and keeping them apart is the point.

**Alignment** is what you are allowed to fit before scoring. A metric model
claims its output is in metres, so fitting anything to the ground truth would
be marking its own homework. A relative model claims only ordering, so it can
only be scored after a scale and shift are fitted -- and the fitted values are
part of the result, not a nuisance to be hidden. Which policy applies is a
property of the model, declared in its spec.json, never guessed from how the
numbers happen to look.

**Metrics** are then the usual three. AbsRel and delta1 are scale-sensitive by
construction, which is exactly why the alignment policy has to be stated
before the number means anything.

Nothing here loads a model or touches a GPU, so it can be tested against
arrays whose answers are known by hand.
"""

import numpy as np

# Fitted on disparity rather than depth for the scale+shift case. That is the
# convention the relative-depth models were trained and published against
# (MiDaS and its descendants), and fitting in depth instead gives visibly
# different numbers for the same prediction -- so the choice has to match the
# model's own, not be picked for convenience.
POLICIES = ("none", "scale", "scale_shift")


def align(pred, gt, mask, policy):
    """Return (aligned prediction, fitted parameters).

    `pred` and `gt` are depth in metres, `mask` marks the pixels with a real
    measurement behind them. The prediction is never modified outside the mask
    fit -- the returned array covers the whole frame so it can still be shown.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown alignment policy {policy!r}; expected one of {POLICIES}")
    m = np.asarray(mask, dtype=bool)
    if policy == "none":
        return np.asarray(pred, dtype=np.float64), {}
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if m.sum() < 2:
        raise ValueError("alignment needs at least two valid pixels")

    if policy == "scale":
        # Least squares on depth with no offset: the model got the shape right
        # and the unit wrong, which is what a scale-only claim means.
        s = float(np.dot(p[m], g[m]) / max(np.dot(p[m], p[m]), 1e-12))
        return p * s, {"scale": s}

    # scale_shift, fitted on disparity
    valid = m & (g > 0) & np.isfinite(p)
    if valid.sum() < 2:
        raise ValueError("scale_shift needs at least two positive-depth pixels")
    y = 1.0 / g[valid]
    x = p[valid]
    # Solved in closed form rather than with np.linalg.lstsq. Two parameters
    # have an exact solution in four sums, and calling LAPACK for it aborts the
    # process on this machine when torch has already been imported -- a
    # Fatal Python error, no traceback, the whole test run gone. A dependency
    # on a linear algebra backend buys nothing here.
    n = x.size
    sx, sy = x.sum(), y.sum()
    sxx, sxy = float(np.dot(x, x)), float(np.dot(x, y))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        # Every prediction the same value: no slope is determined by the data.
        raise ValueError("scale_shift is underdetermined; the prediction is constant")
    s = (n * sxy - sx * sy) / denom
    t = (sy - s * sx) / n
    disp = p * s + t
    # Below this the fitted disparity has crossed zero or gone negative, which
    # is not a depth. Left as inf rather than clipped to a plausible number, so
    # the metrics see it and the mask decides what to do about it.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(disp > 1e-8, 1.0 / disp, np.inf)
    return out, {"scale": float(s), "shift": float(t)}


def metrics(pred, gt, mask, max_depth=None):
    """AbsRel, RMSE, delta1/2/3 over the valid pixels.

    `max_depth` drops ground truth beyond a range the sensor cannot be trusted
    at. It is a property of the dataset, so it is passed in rather than
    assumed, and the count of pixels actually scored is returned with the
    numbers -- two models scored over different pixel sets are not comparable
    and the only way to notice is to publish the count.
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(p) & (g > 0)
    if max_depth is not None:
        m = m & (g <= max_depth)
    n = int(m.sum())
    if not n:
        return {"n": 0}
    p, g = p[m], g[m]
    p = np.maximum(p, 1e-6)
    ratio = np.maximum(p / g, g / p)
    return {
        "n": n,
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "log10": float(np.mean(np.abs(np.log10(p) - np.log10(g)))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
    }


def policy_for(depth_scale):
    """The alignment a model's output contract earns.

    metric   -> none. It claims metres; scoring it after a fit would measure
                a different claim.
    relative -> scale_shift. It claims ordering only.
    anything else -> None, meaning the model cannot be ranked here until
                someone works out what its numbers mean. `unknown` is a real
                answer for VGGT and StreamVGGT today and must not quietly
                become `scale_shift` because that produces a number.
    """
    return {"metric": "none", "relative": "scale_shift"}.get(depth_scale)
