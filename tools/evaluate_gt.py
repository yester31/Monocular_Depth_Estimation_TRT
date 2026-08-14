"""Score engines against measured depth.

    python evaluate_gt.py --manifest data/eval/diode_indoors.json
    python evaluate_gt.py --manifest ... --check      # adapters only, no GPU
    python evaluate_gt.py --manifest ... depth_pro
    python evaluate_gt.py --manifest ... --scale-only # vggt, streamvggt

The comparison table says how fast each engine is and how closely it matches
the graph it was built from. Neither says whether the depth is right. This
does, against DIODE: real measurements in metres, with a mask saying which
pixels the scanner actually reached.

**The adapters are the risky part, so they are checked against an oracle.**
Getting a model onto ground truth means reproducing its preprocessing within a
recorded numerical tolerance
-- channel order, the divide by 255, whether ImageNet statistics apply, which
interpolation, and what the network's output even means -- and every one of
those is a silent failure. Feed a BGR image to a model trained on RGB and it
still returns a plausible depth map that scores badly, and the conclusion
lands on the model.

So each adapter is first run on data/example.jpg and compared against
reports/inputs/<model>.npy, the exact tensor that model's own onnx2trt.py fed
the engine when it was benchmarked. The comparison records max absolute
difference and accepts only values below 1e-4; zero is reported separately as
exact. An adapter outside that tolerance does not get to score anything.
`--check` runs only that part and needs no GPU.

**Alignment is decided by the model's contract, never by the numbers.** A
model that claims metres is scored in metres. See core/gt.py.

**`--scale-only` is the one exception, and it is an exception you have to
type.** VGGT and StreamVGGT return a normalised coordinate frame -- a depth
map whose unit is one average scene distance -- so there is no metre in the
output to score and no shift to fit either, just a single missing number. That
run writes reports/gt_scale_only.md, a separate table, and cannot touch
reports/gt.md. Rows fitted against the measurement do not belong beside rows
that were not.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core import bench, gt, pointmap, spec as spec_mod, unmeasured  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(ROOT, "reports", "inputs")
OUT_DIR = os.path.join(ROOT, "reports", "gt")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Adapters: one per model, transcribed from its own onnx2trt.py.
#
# `pre` takes the original BGR image and reproduces what that model feeds its
# engine, checked against a recorded tensor with max absolute difference below
# 1e-4. `depth` takes the engine's raw outputs and the original size and returns
# depth in metres on the original pixel grid.
# --------------------------------------------------------------------------

def _imagenet_stretch(bgr, h, w):
    """Stretch to the engine's size, then ImageNet mean and standard deviation."""
    img = cv2.resize(bgr, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None]).astype(np.float32)


def _direct_depth(outs, shape, orig_h, orig_w):
    """The output is depth already; put it back on the original grid."""
    d = np.asarray(outs[0]).reshape(shape).astype(np.float64)
    d = cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(d, 1e-3, 1e3)


def _depth_pro_pre(bgr, size=1536):
    """Normalise first, resize second -- the order upstream uses.

    Both steps are linear, so the order does not change the values much, but
    it changes them, and the oracle is byte-exact. torch's bilinear with
    align_corners=False is not cv2's, either.
    """
    import torch

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
    x = ((x - 0.5) / 0.5).unsqueeze(0)
    if x.shape[-2:] != (size, size):
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                            align_corners=False)
    return np.ascontiguousarray(x.numpy())


def _depth_pro_depth(outs, shape, orig_h, orig_w):
    """canonical inverse depth * (W / f_px), then inverted.

    f_px comes from the field of view the model predicts, so nothing per-image
    has to be supplied -- and W is the *original* width, not the network's,
    which is what makes the metres metres.
    """
    canonical = np.asarray(outs[0]).reshape(shape).astype(np.float64)
    fov_deg = float(np.asarray(outs[1]).reshape(-1)[0])
    f_px = 0.5 * orig_w / np.tan(0.5 * np.deg2rad(fov_deg))
    inv = canonical * (orig_w / f_px)
    inv = cv2.resize(inv, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return 1.0 / np.clip(inv, 1e-4, 1e4)



def _point_channel_depth(outs, shape, orig_h, orig_w):
    """Z of a (1, 3, H, W) point map, already in metres. No shift to solve.

    Unlike moge_2 these place the scene themselves -- the point map is in
    camera space -- so the last channel is depth as it stands.
    """
    h, w = shape
    pts = np.asarray(outs[0]).reshape(3, h, w).astype(np.float64)
    d = cv2.resize(pts[2], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(d, 1e-3, 1e3)


PAD_VALUE = [123.675, 116.28, 103.53]
METRIC3D_SIZE = (616, 1064)


def _metric3d_geometry(oh, ow, size=METRIC3D_SIZE):
    """The resize factor and padding metric3d_v2 applies, given a source size."""
    scale = min(size[0] / oh, size[1] / ow)
    rh, rw = int(oh * scale), int(ow * scale)
    pad_h, pad_w = size[0] - rh, size[1] - rw
    return scale, (rh, rw), (pad_h // 2, pad_h - pad_h // 2,
                             pad_w // 2, pad_w - pad_w // 2)


def _metric3d_pre(bgr, size=METRIC3D_SIZE):
    """Keep-ratio resize, then pad, and no normalisation at all.

    The mean and standard deviation are baked into the ONNX graph -- upstream's
    exported model normalises as its first operation -- so it wants raw 0-255.
    Applying ImageNet statistics here would apply them twice.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    oh, ow = rgb.shape[:2]
    _, (rh, rw), pad = _metric3d_geometry(oh, ow, size)
    img = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, pad[0], pad[1], pad[2], pad[3],
                             cv2.BORDER_CONSTANT, value=PAD_VALUE)
    return np.ascontiguousarray(
        img.transpose(2, 0, 1)[None]).astype(np.float32)


def _metric3d_depth(outs, oh, ow, ctx, size=METRIC3D_SIZE):
    """Canonical depth -> metres, which needs the camera that took the picture.

    The network predicts as if through a canonical camera of focal length
    1000 px, so metres are canonical * real_focal * resize_scale / 1000. That
    focal is a property of the image, not of the model, which is why this model
    was recorded as unrankable -- correctly, for an arbitrary photo. A dataset
    that ships one global calibration supplies it, and the manifest carries it.
    """
    fx = ((ctx or {}).get("source", {}).get("intrinsics", {}) or {}).get("fx")
    if not fx:
        raise ValueError(
            "metric3d_v2 needs the real focal length in pixels to produce "
            "metres; the manifest carries no source.intrinsics.fx")
    d = np.asarray(outs[0]).reshape(size).astype(np.float64)
    scale, _, pad = _metric3d_geometry(oh, ow, size)
    d = d[pad[0]:d.shape[0] - pad[1], pad[2]:d.shape[1] - pad[3]]
    d = cv2.resize(d, (ow, oh), interpolation=cv2.INTER_LINEAR)
    # Multiply first, then clamp the metres -- the order upstream Metric3D uses.
    # Clipping canonical at 300 and scaling by ~0.71 put the effective ceiling at
    # 213.4 m instead of 300. No published number moves (DIODE indoors tops out
    # near 17 m) but the two orderings are not the same operation.
    return np.clip(d * (fx * scale / 1000.0), 0, 300)


def _rgb_over_255(bgr, h, w, interp=cv2.INTER_LINEAR):
    """BGR -> RGB, resize, /255. No ImageNet statistics."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (w, h), interpolation=interp).astype(np.float64) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None]).astype(np.float32)



def _square_pad_geometry(oh, ow, net):
    """Where the original image lands inside the padded square, on the net grid.

    Transcribed from models/vggt/onnx2trt.py and models/streamvggt/onnx2trt.py,
    which compute it identically. `scale` is net/max_dim, not net/padded_width:
    copyMakeBorder is given `top, top, left, left`, so when max_dim - width is
    odd the padded image is one pixel short of square and the two disagree by
    under a fifth of a percent. Reproducing the model's own arithmetic matters
    more than removing that, because these coordinates have to name the same
    pixels the network's output does. DIODE is 768x1024, where the difference
    is exactly zero.
    """
    max_dim = max(ow, oh)
    left, top = (max_dim - ow) // 2, (max_dim - oh) // 2
    s = net / max_dim
    return left * s, top * s, (left + ow) * s, (top + oh) * s


def _square_pad_pre(bgr, h, w):
    """Pad to a square with white, resize once, /255. Shape (1, 1, 3, H, W).

    White, not black: upstream's load_fn.py pads with 1.0 on a 0-1 tensor, and
    what fills the padding is part of what the network sees. The rank-5 shape is
    the sequence dimension -- these are multi-view models being fed one view.

    Transcribed rather than imported, like every other adapter here: what binds
    it to the model is reports/inputs/, not a shared function. core/preprocess.py
    is converging on the same arithmetic for the model scripts, and once it has
    settled these two can be collapsed into it -- with the oracle check as the
    thing that proves the collapse was lossless.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    oh, ow = rgb.shape[:2]
    max_dim = max(ow, oh)
    left, top = (max_dim - ow) // 2, (max_dim - oh) // 2
    img = cv2.copyMakeBorder(rgb, top, top, left, left,
                             cv2.BORDER_CONSTANT, value=[255, 255, 255])
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    # onnx2trt.py does this through torch: from_numpy(...).float() / 255.0.
    # The arithmetic is the same IEEE division either way, and the check
    # against reports/inputs/ is what settles it rather than the argument --
    # it comes out at max|diff| 0.0, so this needs no torch on the eval host.
    return np.ascontiguousarray(
        img.transpose(2, 0, 1)[None, None]).astype(np.float32) / 255.0


def _square_pad_depth(outs, shape, orig_h, orig_w):
    """Drop the padding the model was fed, then back onto the original grid.

    The output covers the padded square, so the letterbox has to come off
    before anything is compared to a measurement -- the padding is white paper
    the network invented a depth for. The coordinates are already on the
    output's own grid, so no rescaling stands between them and the crop.

    The values stay in the model's own unit. Nothing here converts them to
    metres, because nothing can: what comes back is a normalised coordinate
    frame and the scale is fitted later, once, in core/gt.scale_only.
    """
    h, w = shape
    d = np.asarray(outs[0]).reshape(h, w).astype(np.float64)
    x1, y1, x2, y2 = _square_pad_geometry(orig_h, orig_w, w)
    d = d[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))]
    d = cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    # Clipped like the metric adapters, but at the bottom only against a zero
    # or negative depth, which is not a depth. There is no upper bound worth
    # asserting in a unit whose size is not known yet.
    return np.where(d > 1e-6, d, np.nan)


def _point_map_depth(outs, shape, orig_h, orig_w, mask_idx, scale_idx):
    """Point-map Z, shifted and scaled into metres, on the original grid.

    The shift comes from core/pointmap.py rather than from the model's clone,
    so this runs on a machine that has only engines. The two agree to 1e-7 on
    real point maps -- see tools/check_pointmap.py, which is what earns the
    right to say "rather than".

    The mask the model returns is used for the solve, because the shift is
    fitted to the geometry and a pixel the model calls invalid has no geometry.
    It is deliberately *not* used to drop pixels from the score: the evaluation
    has its own mask from the scanner, and letting a model mark its own pixels
    invalid would let it discard the ones it finds hard.
    """
    h, w = shape
    points = np.asarray(outs[0]).reshape(h, w, 3)
    mask = np.asarray(outs[mask_idx]).reshape(h, w) > 0.5
    scale = np.asarray(outs[scale_idx]).reshape(-1)[0]
    depth = pointmap.depth_from_point_map(points, mask, scale)
    d = cv2.resize(depth.astype(np.float64), (orig_w, orig_h),
                   interpolation=cv2.INTER_LINEAR)
    return np.where(d > 0, d, np.nan)


# Every adapter is called with the size the benchmark record says the engine was
# built for, never a constant. A hardcoded 518 kept working after
# depth_anything_v2 moved to 672x896: it would have reshaped the output onto the
# wrong grid and scored the model on a scrambled depth map.
ADAPTERS = {
    "depth_anything_v2": {
        "pre": lambda bgr, h, w: _imagenet_stretch(bgr, h, w),
        "depth": lambda outs, h, w, oh, ow, ctx: _direct_depth(outs, (h, w), oh, ow),
    },
    "depth_anything_v3": {
        "pre": lambda bgr, h, w: _imagenet_stretch(bgr, h, w),
        "depth": lambda outs, h, w, oh, ow, ctx: _direct_depth(outs, (h, w), oh, ow),
    },
    "depth_anything_ac": {
        "pre": lambda bgr, h, w: _imagenet_stretch(bgr, h, w),
        "depth": lambda outs, h, w, oh, ow, ctx: _direct_depth(outs, (h, w), oh, ow),
    },
    "distill_any_depth": {
        "pre": lambda bgr, h, w: _imagenet_stretch(bgr, h, w),
        "depth": lambda outs, h, w, oh, ow, ctx: _direct_depth(outs, (h, w), oh, ow),
    },
    "depth_pro": {
        "pre": lambda bgr, h, w: _depth_pro_pre(bgr, h),
        "depth": lambda outs, h, w, oh, ow, ctx: _depth_pro_depth(outs, (h, w), oh, ow),
    },
    "zipdepth": {
        "pre": lambda bgr, h, w: _rgb_over_255(bgr, h, w, cv2.INTER_AREA),
        "depth": lambda outs, h, w, oh, ow, ctx: _direct_depth(outs, (h, w), oh, ow),
    },
    "unidepth_v2": {
        "pre": lambda bgr, h, w: _imagenet_stretch(bgr, h, w),
        "depth": lambda outs, h, w, oh, ow, ctx: _point_channel_depth(outs, (h, w), oh, ow),
    },
    "unik3d": {
        "pre": lambda bgr, h, w: _imagenet_stretch(bgr, h, w),
        "depth": lambda outs, h, w, oh, ow, ctx: _point_channel_depth(outs, (h, w), oh, ow),
    },
    "metric3d_v2": {
        "pre": lambda bgr, h, w: _metric3d_pre(bgr, (h, w)),
        "depth": lambda outs, h, w, oh, ow, ctx: _metric3d_depth(outs, oh, ow, ctx, (h, w)),
    },
    # points, mask, metric_scale -- no normal branch
    "metric_anything": {
        "pre": lambda bgr, h, w: _rgb_over_255(bgr, h, w, cv2.INTER_AREA),
        "depth": lambda outs, h, w, oh, ow, ctx: _point_map_depth(outs, (h, w), oh, ow, 1, 2),
    },
    # points, normal, mask, metric_scale
    "moge_2": {
        "pre": lambda bgr, h, w: _rgb_over_255(bgr, h, w),
        "depth": lambda outs, h, w, oh, ow, ctx: _point_map_depth(outs, (h, w), oh, ow, 2, 3),
    },
    # Normalised coordinate frames -- see NORMALISED_COORDINATES below. Having
    # an adapter is not having a policy: without --scale-only these two report
    # that their output contract earns no alignment, which is the same answer
    # they gave before the adapters existed.
    "vggt": {
        "pre": lambda bgr, h, w: _square_pad_pre(bgr, h, w),
        "depth": lambda outs, h, w, oh, ow, ctx: _square_pad_depth(outs, (h, w), oh, ow),
    },
    "streamvggt": {
        "pre": lambda bgr, h, w: _square_pad_pre(bgr, h, w),
        "depth": lambda outs, h, w, oh, ow, ctx: _square_pad_depth(outs, (h, w), oh, ow),
    },
}


# Why each of these is allowed a scale-only fit, in one sentence that has to be
# defensible. Printed into the report, so the reason travels with the number.
NORMALISED_COORDINATES = {
    "vggt":
        "VGGT normalises its training data by the average Euclidean distance "
        "of the point map to the origin and trains the network to reproduce "
        "that normalisation rather than applying it afterwards "
        "(arXiv:2503.11651, paper body read 2026-08-14), so the output's unit "
        "is one average scene distance and exactly one number is missing.",
    "streamvggt":
        "StreamVGGT is distilled from VGGT and keeps its depth head and its "
        "first-frame reference frame, so it inherits the same normalised unit. "
        "Its own paper (arXiv:2507.11539, read 2026-08-14) does not restate "
        "the normalisation, so this is inferred from the distillation setup "
        "and not quoted from the paper body -- the fitted scales, reported "
        "per model, are what would show it if the inference were wrong.",
}


def check_adapter(model, example_bgr, size):
    """Does this adapter reproduce the tensor the benchmark actually fed?"""
    oracle_path = os.path.join(INPUTS, f"{model}.npy")
    if not os.path.exists(oracle_path):
        return False, f"no {os.path.relpath(oracle_path, ROOT)} to check against"
    oracle = np.load(oracle_path)
    got = ADAPTERS[model]["pre"](example_bgr, *size)
    if got.shape != oracle.shape:
        return False, f"shape {got.shape} against the recorded {oracle.shape}"
    diff = float(np.abs(got.astype(np.float64) - oracle.astype(np.float64)).max())
    # Not exact equality: the recorded tensor went through float32 arithmetic in
    # a different order. A mismatch from a wrong channel order or a missing
    # normalisation is orders of magnitude larger than this.
    return diff < 1e-4, f"max|diff| {diff:.3e}"


def pick_runs(records, want_size=None):
    """model -> the one benchmark record to score, by input size when asked.

    A model can have more than one engine on disk -- that is the whole point of
    a size-sensitivity experiment -- and a dict built by iterating the files
    silently keeps whichever came last. That is not a choice anyone made. When
    several records exist and no size is given, none is returned for that model
    and the run says so.
    """
    by_model = {}
    for r in records:
        if r.get("variant", "single") != "single":
            continue
        by_model.setdefault(r["model"], []).append(r)
    out = {}
    for model, rs in by_model.items():
        if want_size:
            rs = [r for r in rs
                  if f"{r.get('input_h')}x{r.get('input_w')}" == want_size]
        if len(rs) == 1:
            out[model] = rs[0]
        elif len(rs) > 1:
            have = ", ".join(sorted(f"{r.get('input_h')}x{r.get('input_w')}"
                                    for r in rs))
            print(f"  {model}: {len(rs)} records ({have}); pass --size to choose")
    return out


def load_engine(path):
    import tensorrt as trt
    with open(path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.ERROR)) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"could not deserialize {path}")
    return engine


def uint8_engine(run):
    """The uint8-NHWC engine tools/retired/ab_input_dtype.py built beside the float32 one.

    Its filename convention is that tool's, not tune_build's: `_u8` goes in
    before the precision. Reproduced rather than searched for, so a missing
    engine is missing and not silently a different one.
    """
    path = run.get("engine_path") or ""
    stem, ext = os.path.splitext(os.path.basename(path))
    prec = run.get("precision", "fp16")
    if not stem.endswith("_" + prec):
        return None
    candidate = os.path.join(os.path.dirname(path),
                             stem[:-len(prec)] + "u8_" + prec + ext)
    return candidate if os.path.exists(candidate) else None


def _uint8_pre(bgr, h, w):
    """What the graph's own preamble expects: uint8 NHWC, RGB, unnormalised.

    The cast, the divide by 255, the mean and standard deviation and the
    transpose are all inside the engine now. Doing any of them here would do
    them twice.
    """
    img = cv2.resize(bgr, (w, h))
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None]


def tune_engine(run, precision):
    """The engine tune_build wrote for this model at another precision.

    P4 needs the same model scored twice, and the second engine deliberately
    does not live where the first one does -- tune_build renames it so nothing
    resolving the published engine by filename can pick it up. That renaming is
    reproduced here rather than searched for, so a missing engine is a missing
    engine and not a silently different one.
    """
    path = run.get("engine_path") or ""
    stem, ext = os.path.splitext(os.path.basename(path))
    candidate = os.path.join(os.path.dirname(path), "tune",
                             f"{stem}_as_{precision}_optd_wsd{ext}")
    return candidate if os.path.exists(candidate) else None


def alignment_policy(model, sp, scale_only=False):
    """Which alignment this run is entitled to, and why if the answer is none.

    Returns (policy, reason). `policy` is None when nothing is entitled, and
    then `reason` says what would have to change. The declared contract always
    wins: if a spec.json ever says `metric` or `relative`, --scale-only does not
    override it, because at that point the model is claiming something a scale
    fit would quietly measure around.
    """
    declared = gt.policy_for(sp.get("depth_scale"))
    if declared is not None:
        if scale_only:
            return None, (f"spec.json declares {sp.get('depth_scale')!r}, which "
                          f"earns {declared}; --scale-only would score a claim "
                          f"this model is not making")
        return declared, ""
    normalised = gt.policy_for_normalised(sp.get("depth_scale"))
    if normalised is not None:
        return normalised, ""
    if not scale_only:
        return None, "no alignment policy for this output contract"
    if model not in NORMALISED_COORDINATES:
        return None, ("--scale-only is for a normalised coordinate frame and "
                      f"nothing establishes that {model} has one; the models "
                      f"that do are " + ", ".join(sorted(NORMALISED_COORDINATES)))
    return "scale", ""


def score_model(model, manifest, root, runs, limit=None, diagnose=False,
                precision=None, variant=None, scale_only=False):
    from core import common

    run = runs[model]
    engine_path = run["engine_path"]
    eh, ew = run.get("input_h"), run.get("input_w")
    if not eh or not ew:
        return {"model": model, "skip": "benchmark record has no input size"}
    pre = lambda bgr, h=eh, w=ew: ADAPTERS[model]["pre"](bgr, h, w)   # noqa: E731
    if variant == "uint8":
        engine_path = uint8_engine(run)
        if not engine_path:
            return {"model": model,
                    "skip": "no uint8 engine built; run tools/retired/ab_input_dtype.py " + model}
        pre = lambda bgr, h=eh, w=ew: _uint8_pre(bgr, h, w)   # noqa: E731
    elif precision and precision != run.get("precision"):
        engine_path = tune_engine(run, precision)
        if not engine_path:
            return {"model": model,
                    "skip": f"no {precision} engine built; run "
                            f"tools/tune_build.py {model} --precision {precision}"}
    engine = load_engine(engine_path)
    sp = spec_mod.load_all()[model]
    policy, why = alignment_policy(model, sp, scale_only)
    if policy is None:
        return {"model": model, "skip": why}
    # Which way round a relative output runs decides whether the scale+shift fit
    # is affine in the space the model was trained in. There is no safe default,
    # so it is required rather than guessed, and checked against the data below.
    form = sp.get("output_form")
    # Undeclared is not scored -- but it is measured, so the run tells you what
    # to declare instead of leaving you to guess the thing it just refused to
    # guess for you. A few images settle a sign.
    orientation_only = policy == "scale_shift" and form not in ("depth", "inverse_depth")
    if orientation_only:
        limit = min(limit or 5, 5)
    pred_is_inverse = form == "inverse_depth"

    per_sample, fits = [], []
    with engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        samples = manifest["samples"][:limit] if limit else manifest["samples"]
        for s in samples:
            bgr = cv2.imread(os.path.join(root, s["rgb"]))
            if bgr is None:
                continue
            oh, ow = bgr.shape[:2]
            inputs[0].host = pre(bgr)
            raw = common.do_inference(context, engine=engine, bindings=bindings,
                                      inputs=inputs, outputs=outputs, stream=stream)
            pred = ADAPTERS[model]["depth"](raw, eh, ew, oh, ow, manifest)

            g = np.load(os.path.join(root, s["depth"]))
            m = np.load(os.path.join(root, s["mask"])).astype(bool)
            g = g[..., 0] if g.ndim == 3 else g
            m = m[..., 0] if m.ndim == 3 else m
            m = m & (g > 0)
            if orientation_only:
                per_sample.append({"id": s["id"], "n": 0,
                                   "orientation": gt.orientation(pred, g, m)})
                continue
            try:
                # Routed through the named entry point rather than through
                # align("scale"), so the scale-only path is the one thing that
                # had to be asked for and the one thing the tests exercise.
                if policy == "scale":
                    # The scale fit is two dot products over the mask with no
                    # filtering of its own, so one non-finite prediction makes
                    # the fitted scale NaN and every metric with it. These
                    # adapters do produce non-finite values -- a pixel the
                    # model gave a zero or negative depth is not a depth and is
                    # marked, not clipped to something plausible. metrics()
                    # drops the same pixels, so fitting on them and scoring
                    # without them would fit a unit to pixels that were never
                    # scored. Narrowed here rather than inside align(), which
                    # eleven published scores stand on.
                    aligned, fit = gt.scale_only(pred, g, m & np.isfinite(pred))
                else:
                    aligned, fit = gt.align(pred, g, m, policy, pred_is_inverse)
            except ValueError as e:
                per_sample.append({"id": s["id"], "skip": str(e)})
                continue
            r = gt.metrics(aligned, g, m, max_depth=manifest.get("max_depth"))
            r["id"] = s["id"]
            r["orientation"] = gt.orientation(pred, g, m)
            if policy == "scale":
                # Kept per image, not only as a median. A normalised model that
                # returns a different unit on every frame is not producing a
                # unit at all, and the averaged AbsRel would not show it --
                # each image gets its own fit, so each image would still score
                # well. Added on this path alone so the eleven published
                # artefacts keep the shape they were written with.
                r["fit"] = fit
            if diagnose:
                # What the score would be if the model were allowed a fit it
                # does not claim. This is never the published number -- it is
                # the question "is this a wrong scale or a wrong shape?", and
                # the two look identical in AbsRel alone. A metric model whose
                # error collapses under a scale fit is mis-scaled, not blind.
                for alt in ("scale", "scale_shift"):
                    if alt == policy:
                        continue
                    try:
                        a2, f2 = gt.align(pred, g, m, alt, pred_is_inverse)
                    except ValueError:
                        continue
                    r[f"diag_{alt}"] = gt.metrics(
                        a2, g, m, max_depth=manifest.get("max_depth"))
                    r[f"diag_{alt}_fit"] = f2
            per_sample.append(r)
            if fit:
                fits.append(fit)
        common.free_buffers(inputs, outputs, stream)

    if orientation_only:
        seen = [r["orientation"] for r in per_sample
                if r.get("orientation") == r.get("orientation")]
        if not seen:
            return {"model": model, "skip": "output_form undeclared and unmeasurable"}
        med = float(np.median(seen))
        measured = "depth" if med > 0 else "inverse_depth"
        return {"model": model, "orientation_median": med,
                "output_form_measured": measured,
                "skip": f"output_form undeclared. Measured over {len(seen)} images: "
                        f"correlation {med:+.3f} with true depth, so it is "
                        f"'{measured}'. Put that in models/{model}/spec.json."}

    scored = [r for r in per_sample if r.get("n")]
    if not scored:
        return {"model": model, "skip": "nothing scored"}
    # Averaged over images, not over pixels: an image with more valid pixels
    # should not count for more than another image.
    agg = {k: float(np.mean([r[k] for r in scored]))
           for k in ("abs_rel", "rmse", "log10", "delta1", "delta2", "delta3")}
    for alt in ("scale", "scale_shift"):
        got = [r[f"diag_{alt}"] for r in scored if f"diag_{alt}" in r]
        if got:
            agg[f"diag_{alt}"] = {k: float(np.mean([g2[k] for g2 in got]))
                                  for k in ("abs_rel", "delta1")}
            fitted = [r[f"diag_{alt}_fit"] for r in scored
                      if f"diag_{alt}_fit" in r]
            if fitted:
                agg[f"diag_{alt}"]["fit_median"] = {
                    k: float(np.median([f[k] for f in fitted])) for k in fitted[0]}
    orients = [r["orientation"] for r in scored
               if r.get("orientation") == r.get("orientation")]
    if orients:
        agg["orientation_median"] = float(np.median(orients))
        # Positive means the prediction rises with true depth, so it is a depth
        # map; negative means it falls, so it is a disparity map. A declaration
        # that disagrees with the data is reported rather than obeyed.
        measured = "depth" if agg["orientation_median"] > 0 else "inverse_depth"
        agg["output_form_declared"] = form
        agg["output_form_measured"] = measured
        if form and measured != form:
            agg["output_form_conflict"] = (
                f"spec.json says {form} but the prediction correlates "
                f"{agg['orientation_median']:+.3f} with true depth")
    agg.update(model=model, images=len(scored),
               pixels=int(sum(r["n"] for r in scored)),
               alignment=policy, engine_path=engine_path,
               precision=precision or run.get("precision"), samples=per_sample)
    if policy == "scale":
        # The reason travels with the number, in the JSON as well as the table.
        agg["normalised_because"] = NORMALISED_COORDINATES.get(
            model, f"spec.json declares depth_scale {sp.get('depth_scale')!r}")
    if fits:
        agg["fit_median"] = {k: float(np.median([f[k] for f in fits]))
                             for k in fits[0]}
    if policy == "scale" and fits:
        # The spread across images, not just the middle of it. This is the one
        # number that can say "the unit is not consistent", which is a claim
        # about the model that no AbsRel computed after a per-image fit can
        # make -- each image would still score well on its own scale.
        ss = sorted(f["scale"] for f in fits)
        med = agg["fit_median"]["scale"]
        agg["scale_spread"] = {
            "min": ss[0], "max": ss[-1],
            "relative_range": float((ss[-1] - ss[0]) / med) if med else float("nan")}
    return agg


def conditions(r):
    """(h, w, precision, input dtype) the row was measured under.

    Execution rule 8 -- do not let models measured at different input sizes read
    as one ranking -- and this table used to carry neither the size nor the
    precision. `reports/gt.md`, `gt_fp32.md` and `gt_uint8.md` were byte-for-byte
    identical in their headings, so three different sets of conditions looked
    like one table read three times.

    Recorded fields win. The engine filename is the fallback, because the
    artefacts already in reports/gt/ were written before score_model recorded
    the size and the variant, and re-running them needs the GPU box and the
    DIODE set. It is a parse of a name, so it is only ever used to fill a gap:
    tests/test_scale_only.py checks the two agree wherever both exist.
    """
    engine = os.path.basename(str(r.get("engine_path") or ""))
    h, w = r.get("input_h"), r.get("input_w")
    if not (h and w):
        m = re.search(r"(\d+)x(\d+)", engine)
        if m:
            h, w = int(m.group(1)), int(m.group(2))
    dtype = r.get("input_dtype")
    if dtype is None:
        # tools/retired/ab_input_dtype.py names the uint8-NHWC engines with a _u8_ segment.
        dtype = "uint8" if re.search(r"_u8[_.]", engine) else "float32"
    return h, w, r.get("precision"), dtype


def _conditions_section(rows, suffix=""):
    """Engine and result-JSON path for every scored row.

    Execution rule 7 says a number in a table comes out of the JSON rather than
    off a keyboard; this is the other half of that -- which JSON, and which
    engine file. The engine name is also where the input dtype lives: the
    uint8-NHWC engines tools/retired/ab_input_dtype.py builds carry a `_u8_` segment, and the
    `_as_fp32_` ones were rebuilt from an fp16-typed ONNX by tune_build.py.
    """
    L = ["## Conditions", "",
         "Every row above, with the engine it was measured on and the result "
         "file its numbers came from. Two rows are comparable only where these "
         "agree.", ""]
    head = ["model", "input", "precision", "input dtype", "engine", "result JSON"]
    cells = []
    for r in sorted(rows, key=lambda r: r["model"]):
        h, w, precision, dtype = conditions(r)
        cells.append([
            f"`{r['model']}`",
            "--" if not (h and w) else f"{h}x{w}",
            precision or "--",
            dtype,
            f"`{os.path.basename(str(r.get('engine_path') or '--'))}`",
            f"`reports/gt/{r['model']}{suffix}.json`"])
    wid = [max(len(str(c)) for c in [head[i]] + [row[i] for row in cells])
           for i in range(len(head))]
    L.append("| " + " | ".join(h.ljust(x) for h, x in zip(head, wid)) + " |")
    L.append("| " + " | ".join("---" for _ in head) + " |")
    for row in cells:
        L.append("| " + " | ".join(str(c).ljust(x) for c, x in zip(row, wid)) + " |")
    L.append("")
    return L


def render(results, manifest, scale_only=False, suffix=""):
    # A model with no score is a record in reports/gt/ like any other, and it
    # has to leave the row set before anything below asks it for an AbsRel.
    # Three of the fourteen sit here: two whose output is not in metres and one
    # whose evaluation contract does not exist yet. Counting the rows of this
    # table and calling it the model count is what put 14/14 in PLAN.md.
    results, missing = unmeasured.split(results)
    src = manifest["source"]
    title = ("Normalised-coordinate models against ground truth" if scale_only
             else "Depth against ground truth")
    L = [f"# {title}", "",
         f"{src['dataset']} {manifest['split']}, {manifest['count']} images, "
         f"depth in {manifest['depth_units']} as {manifest['depth_convention']}. "
         f"Generated by `evaluate_gt.py`; the image list and its checksums are "
         f"in the manifest.", ""]
    if scale_only:
        L += ["**This is a separate table on purpose. Do not merge it with "
              "`reports/gt.md` and do not read a rank across the two.** Every "
              "row here was multiplied by a number fitted to the measurement "
              "before it was scored; the rows in `gt.md` marked `none` were "
              "not. A model that only had to get the shape right is not "
              "competing with one that had to get the metre right.", ""]
    ok = [r for r in results if not r.get("skip")]
    if ok:
        # input size and precision are conditions, not decoration: the same
        # model at 518x518 and at 672x896 is two measurements, and fp16 against
        # fp32 is two more. They come out of the result JSON, never off a
        # keyboard -- see conditions().
        TEXT_COLS = 4
        head = ["model", "input HxW", "precision", "alignment",
                "AbsRel", "RMSE", "log10", "d1", "d2", "d3", "images"]
        if scale_only:
            head.insert(TEXT_COLS, "fitted scale")
        def row(r):
            h, wd, precision, _ = conditions(r)
            cells = [f"`{r['model']}`",
                     "--" if not (h and wd) else f"{h}x{wd}",
                     precision or "--",
                     r["alignment"]]
            if scale_only:
                # The fitted number is the interesting one here, not a
                # footnote: it is the unit the model was never going to supply.
                s = r.get("fit_median", {}).get("scale")
                cells.append("--" if s is None else f"{s:.3f}")
            return cells + [
                f"{r['abs_rel'] * 100:.2f}%", f"{r['rmse']:.3f}",
                f"{r['log10']:.4f}", f"{r['delta1'] * 100:.1f}%",
                f"{r['delta2'] * 100:.1f}%", f"{r['delta3'] * 100:.1f}%",
                str(r["images"])]

        rows = [row(r) for r in sorted(ok, key=lambda r: r["abs_rel"])]
        w = [max(len(str(c)) for c in [h] + [r[i] for r in rows])
             for i, h in enumerate(head)]
        L.append("| " + " | ".join(h.ljust(x) for h, x in zip(head, w)) + " |")
        L.append("| " + " | ".join("---" if i < TEXT_COLS else "---:"
                                   for i in range(len(head))) + " |")
        for r in rows:
            L.append("| " + " | ".join(str(c).ljust(x) for c, x in zip(r, w)) + " |")
        L += ["", "Best first by AbsRel. Averaged over images rather than over "
                  "pixels, so an image with more valid pixels does not count "
                  "for more than another.", ""]
        L += _conditions_section(ok, suffix)
    skipped = [r for r in results if r.get("skip")]
    if skipped:
        L += ["## Not scored", ""]
        L += [f"- `{r['model']}` -- {r['skip']}" for r in skipped]
        L.append("")
    # "Not scored" above is a scoring run that reached a model and stopped.
    # This is a model the run never had anything to offer -- and the reason it
    # has none is the interesting part, because for two of the three the answer
    # is not "run it later" but "this table is the wrong table".
    L += unmeasured.section(
        missing,
        intro="Fourteen models are in this repository and the table above does "
              "not have fourteen rows. These are the difference. None of them "
              "was dropped for scoring badly.")
    for r in sorted(missing, key=lambda r: r.get("model", "")):
        if r.get("measured_instead"):
            L += [f"`{r['model']}` is measured instead in "
                  f"`{r['measured_instead']}` — "
                  f"{r.get('measured_instead_status', 'status unrecorded')}.", ""]
    L += ["**Alignment** is the model's own claim, not a choice made here. "
          "`none` means the output was compared in metres as it came out; "
          "`scale_shift` means a scale and a shift were fitted on disparity "
          "first, which is all a relative model claims. Two rows with "
          "different alignments are not comparable.", ""]
    if scale_only:
        L += ["`scale` means one number was fitted on depth and nothing else: "
              "the least-squares s in min ||s*pred - gt||^2, which is one dot "
              "product over another. One parameter, because a normalised "
              "output has lost exactly one degree of freedom -- the single "
              "division that put the scene into its own unit. It is not "
              "`scale_shift` with a parameter left out; a shift would be a "
              "correction for an error these models do not have.", ""]
        for r in sorted(results, key=lambda r: r["model"]):
            why = r.get("normalised_because")
            if why:
                L += [f"- `{r['model']}` -- {why}"]
        if any(r.get("normalised_because") for r in results):
            L.append("")
        L += ["The **fitted scale** is a diagnostic and not a score. A model "
              "whose scale wanders from image to image is not producing a "
              "consistent unit at all, which no averaged AbsRel would show; "
              "the per-image values are in `reports/gt/<model>_scale_only.json`.",
              ""]
    return "\n".join(L)


# The manifest header is three facts about the dataset, and re-rendering must
# not invent them. They are read back out of the published file so a rebuild
# cannot silently retitle a table it did not measure.
RERENDER_HEADER = re.compile(
    r"^(?P<dataset>\S+) (?P<split>\S+), (?P<count>\d+) images, "
    r"depth in (?P<units>\S+) as (?P<convention>[^.]+)\.", re.M)

# Which result files feed which table. Anything not listed here is a per-run
# artefact (_scale_only writes its own table from its own suffix).
RERENDER_TABLES = ("", "_fp32", "_uint8", "_scale_only")


def _results_for(suffix):
    """The committed result JSON belonging to one table, in name order."""
    out = []
    if not os.path.isdir(OUT_DIR):
        return out
    others = [s for s in RERENDER_TABLES if s and s != suffix]
    for name in sorted(os.listdir(OUT_DIR)):
        if not name.endswith(".json"):
            continue
        stem = name[:-5]
        if suffix:
            if not stem.endswith(suffix):
                continue
        elif any(stem.endswith(s) for s in others):
            continue
        with open(os.path.join(OUT_DIR, name), encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def rerender():
    """Rebuild the published tables from the JSON already on disk.

    Adding a column to a table whose measurements cost a GPU box, a dataset and
    an afternoon must not mean re-measuring. The scores are read back verbatim
    out of the files the scoring run wrote; only the table around them is built
    again. tests/test_scale_only.py asserts both halves of that -- that the
    published file is what render() produces, and that the numbers in it are
    still the numbers in the JSON.
    """
    wrote = 0
    for suffix in RERENDER_TABLES:
        out = os.path.join(ROOT, "reports", f"gt{suffix}.md")
        if not os.path.exists(out):
            continue
        with open(out, encoding="utf-8") as f:
            published = f.read()
        m = RERENDER_HEADER.search(published)
        if not m:
            print(f"  {os.path.basename(out)}: no dataset header to read back, "
                  f"left alone")
            continue
        # Only the rows the published table already has. reports/gt/ holds the
        # artefacts of every run ever made, and eight *_fp32.json files are on
        # disk against the three rows gt_fp32.md publishes -- those five were
        # scored by later runs that wrote their JSON without republishing the
        # table. Re-rendering is meant to add a column, not to widen the claim,
        # so the row set comes from the file being rebuilt.
        wanted = set(re.findall(r"^\| `([a-z0-9_]+)`", published, re.M))
        # The row set comes from the published file -- except for the records
        # that are not rows. A not-measured record has no score to widen a
        # claim with, and dropping it here because it is absent from the table
        # is exactly the silence it was written to break.
        found = _results_for(suffix)
        results = [r for r in found
                   if r["model"] in wanted or unmeasured.is_unmeasured(r)]
        missing = wanted - {r["model"] for r in results}
        if missing:
            print(f"  {os.path.basename(out)}: no result JSON for "
                  f"{', '.join(sorted(missing))}, left alone")
            continue
        if not results:
            print(f"  {os.path.basename(out)}: no result JSON, left alone")
            continue
        manifest = {"source": {"dataset": m.group("dataset")},
                    "split": m.group("split"), "count": int(m.group("count")),
                    "depth_units": m.group("units"),
                    "depth_convention": m.group("convention")}
        with open(out, "w", encoding="utf-8") as f:
            f.write(render(results, manifest, suffix == "_scale_only", suffix))
        print(f"  rebuilt {os.path.relpath(out, ROOT)} from "
              f"{len(results)} result file(s)")
        wrote += 1
    if not wrote:
        print("nothing to rebuild")
    return 0


def preserve_unmeasured(results, recorded):
    """Keep durable contract records when a refresh can only return a skip."""
    missing = {r.get("model"): r for r in recorded
               if unmeasured.is_unmeasured(r)}
    kept = [missing[r["model"]]
            if r.get("skip") and r.get("model") in missing else r
            for r in results]
    existing = {r["model"] for r in kept}
    kept.extend(r for model, r in missing.items() if model not in existing)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="default: every model with an adapter")
    ap.add_argument("--manifest", default=None,
                    help="required unless --rerender")
    ap.add_argument("--rerender", action="store_true",
                    help="rebuild reports/gt*.md from the result JSON already "
                         "in reports/gt/ and stop. No GPU, no dataset, no "
                         "engine: the numbers are not recomputed, only the "
                         "table around them. This is how a column is added to "
                         "a published table without the scores moving.")
    ap.add_argument("--root", default=None,
                    help="dataset root; default: the manifest's own directory tree")
    ap.add_argument("--limit", type=int, default=None, help="first N images only")
    ap.add_argument("--size", default=None, metavar="HxW",
                    help="pick among several engines of one model by input "
                         "size, e.g. 672x896. Required when more than one "
                         "record exists, because which one wins is otherwise "
                         "whatever order the files were read in")
    ap.add_argument("--variant", default=None, choices=("uint8",),
                    help="score the uint8-NHWC engine instead of the float32 one")
    ap.add_argument("--precision", default=None,
                    help="score the engine tune_build produced at this "
                         "precision instead of the published one")
    ap.add_argument("--diagnose", action="store_true",
                    help="also report what the score would be under an alignment "
                         "the model does not claim, to tell a wrong scale from "
                         "a wrong shape. Never the published number.")
    ap.add_argument("--check", action="store_true",
                    help="verify the adapters against the recorded inputs and stop")
    ap.add_argument("--scale-only", action="store_true",
                    help="score the models whose output is a normalised "
                         "coordinate frame (" +
                         ", ".join(sorted(NORMALISED_COORDINATES)) + ") by "
                         "fitting one scale. Writes reports/gt_scale_only.md, "
                         "a separate table: these rows are not comparable with "
                         "the metric ones and must not be merged into gt.md")
    args = ap.parse_args()

    if args.rerender:
        return rerender()
    if not args.manifest:
        print("--manifest is required (or --rerender to rebuild the tables "
              "from the JSON already on disk)")
        return 2

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    # --scale-only defaults to exactly the models it is for. Defaulting to every
    # adapter instead would run eleven engines to skip eleven of them.
    wanted = args.models or sorted(
        NORMALISED_COORDINATES if args.scale_only else ADAPTERS)
    unknown = [m for m in wanted if m not in ADAPTERS]
    if unknown:
        print(f"no adapter for {', '.join(unknown)} -- "
              f"have {', '.join(sorted(ADAPTERS))}")
        return 2

    sizes = {m: (r.get("input_h"), r.get("input_w")) for m, r in
             pick_runs(bench.load_all(os.path.join(ROOT, "reports", "bench")),
                       args.size).items()}
    example = cv2.imread(os.path.join(ROOT, "data", "example.jpg"))
    if example is None:
        print("data/example.jpg is missing; the adapters cannot be checked")
        return 1

    good = []
    if args.variant == "uint8":
        # The oracle is the float32 tensor, so it cannot check a uint8 adapter.
        # Saying so beats printing a check that did not happen.
        print("uint8 variant: reports/inputs holds the float32 tensor, so the "
              "adapter cannot be checked against it. The graph preamble does "
              "the normalisation; this feeds raw uint8 RGB.")
        good = list(wanted)
    else:
        print("adapter check (against the tensor each benchmark actually fed):")
        for m in wanted:
            sz = sizes.get(m) or (None, None)
            if not all(sz):
                print("  FAIL %-20s no benchmark record with an input size" % m)
                continue
            passed, detail = check_adapter(m, example, sz)
            print(f"  {'ok  ' if passed else 'FAIL'} {m:20} {detail}")
            if passed:
                good.append(m)
    if args.check:
        return 0 if len(good) == len(wanted) else 1
    if not good:
        print("no adapter reproduces its recorded input; nothing scored")
        return 1

    root = args.root or os.path.dirname(os.path.abspath(args.manifest))
    runs = pick_runs(bench.load_all(os.path.join(ROOT, "reports", "bench")),
                     args.size)

    results = []
    for m in good:
        if m not in runs:
            results.append({"model": m, "skip": "not benchmarked"})
            continue
        if not os.path.exists(runs[m].get("engine_path", "")):
            results.append({"model": m, "skip": "engine not on this machine"})
            continue
        print(f"\n[{m}] scoring")
        r = score_model(m, manifest, root, runs, args.limit, args.diagnose,
                        args.precision, args.variant, args.scale_only)
        results.append(r)
        if r.get("skip"):
            print(f"  skipped: {r['skip']}")
        else:
            print(f"  AbsRel {r['abs_rel'] * 100:.2f}%  RMSE {r['rmse']:.3f}  "
                  f"d1 {r['delta1'] * 100:.1f}%  over {r['images']} images")
            if r.get("output_form_conflict"):
                print(f"  !! {r['output_form_conflict']}")
            sp_ = r.get("scale_spread")
            if sp_:
                print(f"    fitted scale median "
                      f"{r['fit_median']['scale']:.4f}, range "
                      f"{sp_['min']:.4f}-{sp_['max']:.4f} "
                      f"({sp_['relative_range'] * 100:.1f}% of the median)")
            for alt in ("scale", "scale_shift"):
                d = r.get(f"diag_{alt}")
                if d:
                    fit = d.get("fit_median", {})
                    extra = " ".join(f"{k}={v:.4g}" for k, v in fit.items())
                    print(f"    if {alt:12} were allowed: AbsRel "
                          f"{d['abs_rel'] * 100:.2f}%  d1 {d['delta1'] * 100:.1f}%"
                          f"   median {extra}")

    # Defined before it is used. It was not, and the run scored five models,
    # printed every number, and died on a NameError before writing any of them.
    suffix = f"_{args.precision}" if args.precision else ""
    if args.variant:
        suffix += "_" + args.variant
    if args.scale_only:
        # A suffix, so this run cannot overwrite reports/gt.md. The eleven
        # published rows are in that file and a scale-fitted row does not
        # belong beside them -- keeping them in separate files is the cheapest
        # way to make merging them a deliberate act rather than an accident.
        suffix += "_scale_only"
    # A targeted run can only produce records for models it knows how to run.
    # Preserve explicit not-measured records for the remaining contracts so a
    # successful refresh cannot make an acknowledged gap disappear from the
    # published report. This is how tr2m stays visible in gt.md even though it
    # has no evaluate_gt adapter.
    # A run that reaches a deliberately unsupported model returns a skip, but
    # the existing record is more informative: it names the contract and the
    # alternative table. Do not replace that durable explanation with a
    # four-word runtime result.
    results = preserve_unmeasured(results, _results_for(suffix))
    os.makedirs(OUT_DIR, exist_ok=True)
    for r in results:
        with open(os.path.join(OUT_DIR, f"{r['model']}{suffix}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(r, f, indent=2)
    out = os.path.join(ROOT, "reports", f"gt{suffix}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(results, manifest, args.scale_only, suffix))
    print(f"\nwrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
