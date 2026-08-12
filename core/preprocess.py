"""Shared preprocessing for every model in this repository.

The 12 models resize their input in six distinct ways and normalise it in four.
Each was previously reimplemented per model, which is how the defects in
docs/model_contracts.md crept in — a missing /255 here, black padding where
upstream pads white there.

This module implements the *types*; each model declares which type and which
parameters it uses. Nothing here is model-specific.

Dependency-free apart from numpy/cv2 on purpose: it must import in every
per-model conda environment as well as the shared TensorRT one.
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


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class Geometry:
    """How the source image was mapped into the network input.

    Postprocessing needs this to undo the mapping — cropping padding away,
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

    'ceil'      always round up          — depth_anything_ac
    'constrain' round to nearest, but fall back to floor above `max_val` and to
                ceil below `min_val`    — depth_anything_v2, distill_any_depth
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

    Used by unidepth_v2 / unik3d / depth_pro in this repo. Note that upstream
    unidepth_v2 and unik3d pad instead — see defect D11.
    """
    h, w = img.shape[:2]
    out = cv2.resize(img, (dst_w, dst_h), interpolation=INTERP[interp])
    return out, Geometry(h, w, dst_h, dst_w, inner_h=dst_h, inner_w=dst_w)


def resize_keep_ratio(img, target, multiple=14, bound="lower", interp="cubic",
                      rounding="constrain"):
    """Scale so the short ('lower') or long ('upper') side hits `target`.

    No padding: the output is whatever size the rule produces, snapped to a
    multiple. depth_anything_v2/ac and depth_anything_v3 work this way, but
    with different `rounding` — see _round_to_multiple.
    """
    h, w = img.shape[:2]
    scale = (target / min(h, w)) if bound == "lower" else (target / max(h, w))
    kw = {"min_val": target} if rounding == "constrain" else {}
    new_h = _round_to_multiple(h * scale, multiple, rounding, **kw)
    new_w = _round_to_multiple(w * scale, multiple, rounding, **kw)
    out = cv2.resize(img, (new_w, new_h), interpolation=INTERP[interp])
    return out, Geometry(h, w, new_h, new_w, inner_h=new_h, inner_w=new_w)


def resize_pad(img, dst_h, dst_w, pad_value=0, interp="cubic", center=True):
    """Scale preserving aspect ratio, then pad to exactly (dst_h, dst_w).

    `pad_value` is a scalar or per-channel sequence in the image's own units
    (0-255 for uint8). vggt/streamvggt upstream pad **white**; metric3d pads
    with the ImageNet mean.
    """
    h, w = img.shape[:2]
    scale = min(dst_h / h, dst_w / w)
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


def resize_square_pad(img, dst, pad_value=0, interp="cubic"):
    """Pad to a square first, then resize to dst x dst.

    Equivalent in result to resize_pad on a square target, but this is the
    formulation vggt/streamvggt use, and it is what their coordinate bookkeeping
    is written against.
    """
    return resize_pad(img, dst, dst, pad_value=pad_value, interp=interp)


def resize_none(img, **_):
    """Caller already produced the right size."""
    h, w = img.shape[:2]
    return img, Geometry(h, w, h, w, inner_h=h, inner_w=w)


RESIZE_TYPES = {
    "stretch": resize_stretch,
    "keep_ratio": resize_keep_ratio,
    "pad": resize_pad,
    "square_pad": resize_square_pad,
    "none": resize_none,
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize(img, mode="imagenet", mean=None, std=None):
    """uint8 HWC image -> float32 HWC.

    'imagenet'  /255 then (x-mean)/std with the ImageNet statistics
    'half'      /255 then (x-0.5)/0.5, i.e. [-1, 1]   (depth_pro)
    'scale'     /255 only                             (moge_2, metric_anything, vggt)
    'custom'    /255 then the supplied mean/std
    'raw255'    no scaling; caller-supplied mean/std in 0-255 units (metric3d_v2)
    'none'      passthrough as float32
    """
    x = img.astype(np.float32)
    if mode == "none":
        return x
    if mode == "raw255":
        if mean is not None:
            x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
        return x

    x /= 255.0
    if mode == "scale":
        return x
    if mode == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif mode == "half":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif mode != "custom":
        raise ValueError(f"unknown normalize mode: {mode}")
    return (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)


def to_nchw(img):
    """HWC float32 -> contiguous 1CHW float32."""
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@dataclass
class PreprocessSpec:
    """One model's preprocessing, as data rather than code."""

    resize: str = "stretch"
    size: Optional[Tuple[int, int]] = None      # (h, w) for stretch / pad
    target: Optional[int] = None                # for keep_ratio / square_pad
    multiple: int = 14
    bound: str = "lower"
    rounding: str = "constrain"
    interp: str = "cubic"
    pad_value: Sequence[float] = (0, 0, 0)
    to_rgb: bool = True
    normalize: str = "imagenet"
    mean: Optional[Sequence[float]] = None
    std: Optional[Sequence[float]] = None
    # Whether the resize runs on the raw uint8 image or after scaling to float.
    # Not cosmetic: cubic interpolation overshoots, and on uint8 the overshoot
    # is clipped to [0,255] while on float it is not. depth_anything_v2/ac and
    # distill_any_depth divide by 255 first; metric3d_v2, vggt, streamvggt and
    # moge_2 resize the uint8 image.
    resize_on: str = "uint8"
    extra: dict = field(default_factory=dict)


def _apply_resize(img, spec):
    fn = RESIZE_TYPES[spec.resize]
    if spec.resize == "stretch":
        return fn(img, *spec.size, interp=spec.interp)
    if spec.resize == "keep_ratio":
        return fn(img, spec.target, multiple=spec.multiple, bound=spec.bound,
                  interp=spec.interp, rounding=spec.rounding)
    if spec.resize == "pad":
        return fn(img, *spec.size, pad_value=spec.pad_value, interp=spec.interp)
    if spec.resize == "square_pad":
        return fn(img, spec.target, pad_value=spec.pad_value, interp=spec.interp)
    return fn(img)


def preprocess(img_bgr, spec: PreprocessSpec):
    """BGR uint8 frame -> (1CHW float32 blob, Geometry).

    Colour conversion happens before resizing so pad_value is expressed in the
    same channel order the network sees.
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if spec.to_rgb else img_bgr

    if spec.resize_on == "float":
        img = normalize(img, spec.normalize, spec.mean, spec.std)
        out, geom = _apply_resize(img, spec)
        return to_nchw(out), geom

    out, geom = _apply_resize(img, spec)
    return to_nchw(normalize(out, spec.normalize, spec.mean, spec.std)), geom
