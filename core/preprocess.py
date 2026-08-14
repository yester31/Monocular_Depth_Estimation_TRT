"""Shared preprocessing for every model in this repository.

The 14 models resize their input in four distinct ways and normalise it in four.
Each was previously reimplemented per model, which is how the defects in
docs/model_contracts.md crept in -- a missing /255 here, black padding where
upstream pads white there.

This module implements the *types*; each model declares which type and which
parameters it uses, in MODELS below. Nothing here is model-specific except that
table, and every entry in it is checked byte-for-byte against
reports/inputs/<model>.npy by tests/test_preprocess.py.

**Byte-exact, not close.** reports/inputs/<model>.npy is the tensor that model's
own onnx2trt.py handed its engine, and it is what verify_accuracy.py replays
through the ONNX graph. If this module produced a tensor that were merely close,
all fourteen accuracy baselines would shift by an unknown amount and no later
comparison would say what it claims to say. So the arithmetic *dtype* of each
step is part of the spec, not an implementation detail:

    uint8 / 255.0                       -> float64  (numpy promotes)
    uint8.astype(np.float32) / 255.0    -> float32  (a Python float does not)
    float32_array - [0.485, ...]        -> float64  (a Python list is float64)

Six models declare `stretch` + `imagenet` in spec.json and no two of the three
combinations above produce the same bytes. See docs/model_contracts.md, "전처리 유형과 산술 dtype".

Dependency-free apart from numpy/cv2 on purpose: it must import in every
per-model conda environment as well as the shared TensorRT one. That rules out
depth_pro, whose resize is torch's bilinear and cannot be reproduced by cv2 to
the last bit -- it keeps its own path, and the measured gap is in the audit note.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INTERP = {
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "nearest": cv2.INTER_NEAREST,
}


class UnsupportedPreprocess(NotImplementedError):
    """This model's preprocessing cannot be expressed here byte-exactly."""


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class Geometry:
    """How the source image was mapped into the network input.

    Postprocessing needs this to undo the mapping -- cropping padding away,
    rescaling intrinsics, or resizing a depth map back to the original frame.
    Preprocessing and postprocessing must be treated as a pair; several of the
    known defects come from them disagreeing.
    """

    src_h: int
    src_w: int
    dst_h: int
    dst_w: int
    # size the image occupies inside the destination, before padding
    inner_h: int = 0
    inner_w: int = 0
    # padding applied to reach (dst_h, dst_w)
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    # The same box in sub-pixel form: (x1, y1, x2, y2) on the destination grid.
    # vggt and streamvggt carry exactly these four floats through to their
    # postprocess and slice with int(round(.)), so the contract is the float,
    # not the rounded integer. None when the mapping lands on whole pixels.
    box: Optional[Tuple[float, float, float, float]] = None

    @property
    def scale_x(self) -> float:
        return self.inner_w / self.src_w

    @property
    def scale_y(self) -> float:
        return self.inner_h / self.src_h

    def crop_box(self) -> Tuple[int, int, int, int]:
        """(y0, y1, x0, x1) of the real image inside a dst-sized output."""
        return (self.pad_top, self.pad_top + self.inner_h,
                self.pad_left, self.pad_left + self.inner_w)

    def unpad(self, arr: np.ndarray) -> np.ndarray:
        """Drop the padded border from a dst-sized map."""
        y0, y1, x0, x1 = self.crop_box()
        return arr[y0:y1, x0:x1, ...]

    def to_source(self, arr: np.ndarray, interp: str = "linear") -> np.ndarray:
        """Unpad, then resize back to the original resolution."""
        cropped = self.unpad(arr)
        if cropped.shape[:2] == (self.src_h, self.src_w):
            return cropped
        return cv2.resize(cropped, (self.src_w, self.src_h), interpolation=INTERP[interp])


def _round_to_multiple(x, multiple, mode="round", min_val=None, max_val=None):
    """Snap `x` to a multiple. The rounding rule is model-specific.

    'ceil'      always round up          -- depth_anything_ac
    'constrain' round to nearest, but fall back to floor above `max_val` and to
                ceil below `min_val`    -- depth_anything_v2, distill_any_depth
                (their constrain_to_multiple_of)
    'floor' / 'round' are the plain forms.

    Getting this wrong changes the network input size: for a 4:3 image at
    target 518, 'ceil' gives 700 and 'constrain' gives 686.
    """
    if multiple <= 1:
        return int(round(x))

    if mode == "constrain":
        y = int(np.round(x / multiple) * multiple)
        if max_val is not None and y > max_val:
            y = int(np.floor(x / multiple) * multiple)
        if min_val is not None and y < min_val:
            y = int(np.ceil(x / multiple) * multiple)
        return max(y, multiple)

    q = x / multiple
    q = np.ceil(q) if mode == "ceil" else np.floor(q) if mode == "floor" else np.round(q)
    return max(int(q) * multiple, multiple)


# ---------------------------------------------------------------------------
# Resize types
# ---------------------------------------------------------------------------

def resize_stretch(img, dst_h, dst_w, interp="linear"):
    """Ignore aspect ratio; fill the target exactly.

    Ten of the fourteen models do this, because a static TensorRT engine has one
    input size and the alternatives -- letterboxing or cropping -- either feed
    the network invented pixels or throw real ones away. Upstream unidepth_v2
    and unik3d pad instead; see defect D11.
    """
    h, w = img.shape[:2]
    out = cv2.resize(img, (dst_w, dst_h), interpolation=INTERP[interp])
    return out, Geometry(h, w, dst_h, dst_w, inner_h=dst_h, inner_w=dst_w)


def resize_keep_ratio(img, target, multiple=14, bound="lower", interp="cubic",
                      rounding="constrain"):
    """Scale so the short ('lower') or long ('upper') side hits `target`.

    No padding: the output is whatever size the rule produces, snapped to a
    multiple. depth_anything_v2/ac and distill_any_depth carry this rule, but
    with different `rounding` -- see _round_to_multiple.
    """
    h, w = img.shape[:2]
    scale = (target / min(h, w)) if bound == "lower" else (target / max(h, w))
    kw = {"min_val": target} if rounding == "constrain" else {}
    new_h = _round_to_multiple(h * scale, multiple, rounding, **kw)
    new_w = _round_to_multiple(w * scale, multiple, rounding, **kw)
    out = cv2.resize(img, (new_w, new_h), interpolation=INTERP[interp])
    return out, Geometry(h, w, new_h, new_w, inner_h=new_h, inner_w=new_w)


def resize_snap(img, dst_h, dst_w, multiple=14, interp="cubic", rounding="constrain"):
    """Stretch to (dst_h, dst_w), each side first snapped to a multiple.

    distill_any_depth's variant: it derives a per-axis scale and then runs both
    through constrain_to_multiple_of, so unlike resize_keep_ratio the aspect
    ratio is not preserved. At 518x518 with multiple 14 the snap is exact and
    this is a plain stretch, which is why it never showed up as a difference.
    """
    h, w = img.shape[:2]
    kw_h = {"min_val": dst_h} if rounding == "constrain" else {}
    kw_w = {"min_val": dst_w} if rounding == "constrain" else {}
    new_h = _round_to_multiple(dst_h, multiple, rounding, **kw_h)
    new_w = _round_to_multiple(dst_w, multiple, rounding, **kw_w)
    out = cv2.resize(img, (new_w, new_h), interpolation=INTERP[interp])
    return out, Geometry(h, w, new_h, new_w, inner_h=new_h, inner_w=new_w)


def resize_pad(img, dst_h, dst_w, pad_value=0, interp="cubic", center=True,
               rounding="round"):
    """Scale preserving aspect ratio, then pad to exactly (dst_h, dst_w).

    `pad_value` is a scalar or per-channel sequence in the image's own units
    (0-255 for uint8). metric3d_v2 pads with the ImageNet mean colour.

    `rounding` is 'trunc' for metric3d_v2, which computes int(w * scale) rather
    than rounding it. On most sizes the two agree; where they do not, the whole
    padded layout shifts by a pixel, so this is a parameter and not a tidy-up.
    """
    h, w = img.shape[:2]
    scale = min(dst_h / h, dst_w / w)
    if rounding == "trunc":
        inner_h, inner_w = int(h * scale), int(w * scale)
    else:
        inner_h, inner_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (inner_w, inner_h), interpolation=INTERP[interp])

    pad_h, pad_w = dst_h - inner_h, dst_w - inner_w
    top = pad_h // 2 if center else 0
    left = pad_w // 2 if center else 0
    bottom, right = pad_h - top, pad_w - left

    value = list(pad_value) if isinstance(pad_value, Sequence) else [pad_value] * 3
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=value)
    return out, Geometry(h, w, dst_h, dst_w, inner_h=inner_h, inner_w=inner_w,
                         pad_top=top, pad_bottom=bottom, pad_left=left, pad_right=right)


def resize_square_pad(img, dst_h, dst_w, pad_value=0, interp="cubic",
                      symmetric=True):
    """Pad to a square at source resolution, *then* resize once.

    This is not resize_pad on a square target: padding first means the
    interpolation sees the pad colour, so the two produce different pixels along
    the seam. vggt and streamvggt do it in this order and their coordinate
    bookkeeping is written against it.

    `symmetric=True` reproduces their border exactly:

        left = (max_dim - w) // 2
        cv2.copyMakeBorder(img, top, top, left, left, ...)

    -- the *same* count on both sides rather than top and (max_dim - h - top).
    When max_dim - h is odd that is one pixel short of square and the resize
    stretches by that pixel. data/example.jpg is 3024x2268, an even difference,
    so the recorded tensor never sees it; a 1025x769 frame would. Reproduced
    rather than corrected, because correcting it moves the tensor.
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)
    left = (max_dim - w) // 2
    top = (max_dim - h) // 2
    right = left if symmetric else max_dim - w - left
    bottom = top if symmetric else max_dim - h - top

    value = list(pad_value) if isinstance(pad_value, Sequence) else [pad_value] * 3
    padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=value)
    out = cv2.resize(padded, (dst_w, dst_h), interpolation=INTERP[interp])

    # Upstream's scale is dst / max_dim -- the square it *meant* to build, not
    # the possibly-one-short square it did build. Kept as-is: these four floats
    # are the contract the postprocess slices with.
    scale = dst_w / max_dim
    x1, y1 = left * scale, top * scale
    x2, y2 = (left + w) * scale, (top + h) * scale
    px, py = int(round(x1)), int(round(y1))
    iw, ih = int(round(x2)) - px, int(round(y2)) - py
    return out, Geometry(h, w, dst_h, dst_w, inner_h=ih, inner_w=iw,
                         pad_top=py, pad_bottom=dst_h - py - ih,
                         pad_left=px, pad_right=dst_w - px - iw,
                         box=(x1, y1, x2, y2))


def resize_none(img, **_):
    """Caller already produced the right size."""
    h, w = img.shape[:2]
    return img, Geometry(h, w, h, w, inner_h=h, inner_w=w)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
#
# Split into its two halves on purpose. The Depth-Anything family divides by
# 255, *then* resizes, *then* subtracts the mean, so a single normalize() call
# cannot express it without changing the order of operations -- and changing the
# order changes the bytes.

# spec.json's vocabulary on the left, this module's on the right.
NORMALISE_ALIASES = {"over_255": "scale"}

_STATS = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}
_NO_DIVIDE = ("none", "raw255")
_NO_STATS = ("scale", "none")


def scale(img, mode="imagenet", dtype="float64"):
    """First half: cast, and divide by 255 unless the mode says not to.

    `dtype` is the dtype the division happens in and it is load-bearing --
    float64 for depth_anything_v2, float32 for depth_anything_ac, from the same
    source image, and the two disagree in the last bits of the result.
    """
    mode = NORMALISE_ALIASES.get(mode, mode)
    x = img.astype(np.dtype(dtype))
    if mode in _NO_DIVIDE:
        return x
    return x / 255.0


def standardize(x, mode="imagenet", mean=None, std=None, dtype="float64"):
    """Second half: (x - mean) / std, in `dtype`.

    'imagenet'  the ImageNet statistics
    'half'      (x - 0.5) / 0.5, i.e. [-1, 1]          (depth_pro)
    'scale'     nothing to do                          (moge_2, vggt, zipdepth)
    'custom'    the supplied mean/std, 0-1 units
    'raw255'    the supplied mean/std, 0-255 units
    'none'      nothing to do                          (metric3d_v2)
    """
    mode = NORMALISE_ALIASES.get(mode, mode)
    if mode in _NO_STATS:
        return x
    if mode in _STATS:
        mean, std = _STATS[mode]
    elif mode not in ("custom", "raw255"):
        raise ValueError(f"unknown normalize mode: {mode}")
    if mean is None or std is None:
        return x
    d = np.dtype(dtype)
    return (x - np.asarray(mean, d)) / np.asarray(std, d)


def normalize(img, mode="imagenet", mean=None, std=None,
              scale_dtype="float64", norm_dtype="float64"):
    """Both halves, for the models that do them back to back."""
    return standardize(scale(img, mode, scale_dtype), mode, mean, std, norm_dtype)


def to_nchw(img, rank=4, dtype="float32"):
    """HWC -> contiguous float32 1CHW, or 11CHW for the rank-5 models."""
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[None]).astype(np.dtype(dtype))
    if rank == 5:
        x = x[None]
    return np.ascontiguousarray(x)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@dataclass
class ResizeStep:
    """One geometry stage, as data rather than code."""

    kind: str = "none"                  # stretch|keep_ratio|snap|pad|square_pad|none
    size: Optional[Tuple[int, int]] = None      # (h, w) for stretch/pad/square_pad/snap
    target: Optional[int] = None                # for keep_ratio
    multiple: int = 14
    bound: str = "lower"
    rounding: str = "constrain"                 # keep_ratio/snap: how to snap
    pad_rounding: str = "round"                 # pad: 'trunc' for metric3d_v2
    pad_symmetric: bool = True                  # square_pad: mirror the near border
    interp: str = "linear"
    pad_value: Sequence[float] = (0, 0, 0)

    def apply(self, img):
        k = self.kind
        if k == "none":
            return resize_none(img)
        if k == "stretch":
            return resize_stretch(img, *self.size, interp=self.interp)
        if k == "keep_ratio":
            return resize_keep_ratio(img, self.target, multiple=self.multiple,
                                     bound=self.bound, interp=self.interp,
                                     rounding=self.rounding)
        if k == "snap":
            return resize_snap(img, *self.size, multiple=self.multiple,
                               interp=self.interp, rounding=self.rounding)
        if k == "pad":
            return resize_pad(img, *self.size, pad_value=self.pad_value,
                              interp=self.interp, rounding=self.pad_rounding)
        if k == "square_pad":
            return resize_square_pad(img, *self.size, pad_value=self.pad_value,
                                     interp=self.interp,
                                     symmetric=self.pad_symmetric)
        raise ValueError(f"unknown resize kind: {k}")


@dataclass
class PreprocessSpec:
    """One model's preprocessing, as data rather than code.

    Two geometry stages because four models have two: main() squashes the frame
    to the engine's size on uint8, and preprocess_image() then runs a keep-ratio
    rule on the scaled float image. At the built profiles that second stage is
    the identity -- cv2.resize to the size the image already has is exact -- but
    it is not the identity at depth_anything_ac's `native` profile, so it is
    modelled rather than dropped.
    """

    resize: ResizeStep = field(default_factory=ResizeStep)
    post_resize: Optional[ResizeStep] = None
    to_rgb: bool = True
    normalize: str = "imagenet"
    mean: Optional[Sequence[float]] = None
    std: Optional[Sequence[float]] = None
    # The dtypes each arithmetic step lands in. See the module docstring.
    scale_dtype: str = "float64"
    norm_dtype: str = "float64"
    rank: int = 4
    extra: dict = field(default_factory=dict)


def preprocess(img_bgr, spec: PreprocessSpec):
    """BGR uint8 frame -> (float32 blob, Geometry).

    Colour conversion happens before the geometry so pad_value is expressed in
    the same channel order the network sees. That is free: cvtColor and
    cv2.resize commute exactly, checked in tests/test_preprocess.py, so the
    models that resize first and the models that convert first agree bit for bit.
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if spec.to_rgb else img_bgr

    img, geom = spec.resize.apply(img)
    x = scale(img, spec.normalize, spec.scale_dtype)
    if spec.post_resize is not None:
        x, geom2 = spec.post_resize.apply(x)
        # The second stage owns the final size; the first owns the source.
        geom = Geometry(geom.src_h, geom.src_w, geom2.dst_h, geom2.dst_w,
                        inner_h=geom2.inner_h, inner_w=geom2.inner_w)
    x = standardize(x, spec.normalize, spec.mean, spec.std, spec.norm_dtype)
    return to_nchw(x, rank=spec.rank), geom


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------
#
# spec.json stays the declaration of record: type, normalise and interpolation
# are read from it and cross-checked against every entry here by
# tests/test_preprocess.py. What lives here is what spec.json has no field for
# -- the arithmetic dtypes, the rounding quirks, the tensor rank -- plus the
# second geometry stage.
#
# Two known gaps between the two, both documented in
# docs/model_contracts.md "전처리 유형과 산술 dtype" rather than papered over:
#   * spec.json calls both `pad` and `square_pad` "keep_ratio_pad";
#   * spec.json's `verified: "byte-exact ..."` describes evaluate_gt's 1e-4
#     adapter check, which is not byte-exact. This module's check is.

_IMAGENET_PAD = (123.675, 116.28, 103.53)


def _stretch(size, interp, normalise, scale_dtype, norm_dtype, rank=4):
    return PreprocessSpec(
        resize=ResizeStep("stretch", size=size, interp=interp),
        normalize=normalise, scale_dtype=scale_dtype, norm_dtype=norm_dtype,
        rank=rank)


def _builders():
    """model -> (h, w) -> PreprocessSpec. A function per model, not a dict of
    dicts, because four of them need a second stage that depends on the size."""

    def da_v2(h, w):
        s = _stretch((h, w), "linear", "imagenet", "float64", "float64")
        s.post_resize = ResizeStep("keep_ratio", target=h, multiple=14,
                                   bound="lower", rounding="constrain",
                                   interp="cubic")
        return s

    def da_ac(h, w, stretch=True):
        s = _stretch((h, w), "linear", "imagenet", "float32", "float64")
        if not stretch:                      # the `native` profile skips it
            s.resize = ResizeStep("none")
        s.post_resize = ResizeStep("keep_ratio", target=h, multiple=14,
                                   bound="lower", rounding="ceil", interp="cubic")
        return s

    def da_v3(h, w):
        return _stretch((h, w), "linear", "imagenet", "float64", "float64")

    def distill(h, w):
        s = _stretch((h, w), "linear", "imagenet", "float64", "float64")
        s.post_resize = ResizeStep("snap", size=(h, w), multiple=14,
                                   rounding="constrain", interp="cubic")
        return s

    def metric3d(h, w):
        return PreprocessSpec(
            resize=ResizeStep("pad", size=(h, w), interp="linear",
                              pad_value=_IMAGENET_PAD, pad_rounding="trunc"),
            normalize="none", scale_dtype="float32", norm_dtype="float32")

    def vggt(h, w):
        return PreprocessSpec(
            resize=ResizeStep("square_pad", size=(h, w), interp="cubic",
                              pad_value=(255, 255, 255), pad_symmetric=True),
            normalize="over_255", scale_dtype="float32", norm_dtype="float32",
            rank=5)

    return {
        "depth_anything_v2": da_v2,
        "depth_anything_v3": da_v3,
        "depth_anything_ac": da_ac,
        "distill_any_depth": distill,
        "metric_anything": lambda h, w: _stretch((h, w), "area", "over_255",
                                                 "float64", "float64"),
        "moge_2": lambda h, w: _stretch((h, w), "linear", "over_255",
                                        "float64", "float64"),
        "zipdepth": lambda h, w: _stretch((h, w), "area", "over_255",
                                          "float32", "float32"),
        "tr2m": lambda h, w: _stretch((h, w), "area", "imagenet",
                                      "float32", "float32"),
        "unidepth_v2": lambda h, w: _stretch((h, w), "linear", "imagenet",
                                             "float32", "float32"),
        "unik3d": lambda h, w: _stretch((h, w), "linear", "imagenet",
                                        "float32", "float32"),
        "metric3d_v2": metric3d,
        "vggt": vggt,
        "streamvggt": vggt,
    }


MODELS = _builders()

# Not a bug and not an oversight: depth_pro normalises before resizing and
# resizes with torch.nn.functional.interpolate(bilinear, align_corners=False).
# cv2's INTER_LINEAR is the same mathematics and still differs from the recorded
# tensor at 1.788e-07 on 34% of its elements, because the two accumulate the
# four taps in a different order. Importing torch here to close that gap would
# cost this module the property it exists for -- importing everywhere.
UNSUPPORTED = {
    "depth_pro": "torch bilinear resize; cv2 differs at 1.8e-07 on 34% of the "
                 "tensor. See docs/findings.md P1 for why depth_pro stayed behind.",
}


def spec_for(model, size, **kw) -> PreprocessSpec:
    """The PreprocessSpec for one model at one input size.

    `size` is (h, w) -- take it from core.spec.size_of(spec) so the engine's
    size and the preprocessing agree by construction rather than by two people
    editing two constants.
    """
    if model in UNSUPPORTED:
        raise UnsupportedPreprocess(f"{model}: {UNSUPPORTED[model]}")
    if model not in MODELS:
        raise KeyError(f"no preprocessing declared for {model!r}")
    return MODELS[model](*size, **kw)


def preprocess_for(img_bgr, model, size, **kw):
    """spec_for() then preprocess(), for the one-line call sites."""
    return preprocess(img_bgr, spec_for(model, size, **kw))
