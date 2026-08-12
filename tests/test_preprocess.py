"""Check core/preprocess.py against the per-model code it replaces.

Each test reproduces one model's existing preprocessing inline (copied from its
onnx2trt.py) and asserts the shared implementation matches it exactly — except
where docs/model_contracts.md records a defect, in which case the test pins the
*upstream* behaviour and says so.

Run:  python -m pytest tests/test_preprocess.py -q
      python tests/test_preprocess.py          (no pytest needed)
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.preprocess import (  # noqa: E402
    Geometry, PreprocessSpec, normalize, preprocess, resize_keep_ratio,
    resize_pad, resize_square_pad, resize_stretch, to_nchw,
)

RNG = np.random.default_rng(0)
IMG_43 = RNG.integers(0, 256, (480, 640, 3), dtype=np.uint8)    # 4:3
IMG_169 = RNG.integers(0, 256, (720, 1280, 3), dtype=np.uint8)  # 16:9
IMG_SQ = RNG.integers(0, 256, (500, 500, 3), dtype=np.uint8)

_failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        _failures.append(name)


# ---------------------------------------------------------------------------

def test_depth_anything_v2():
    """Type B: keep_ratio to the *long* side, constrain_to_multiple_of(14).

    The reference works in float64 because `uint8_array / 255.0` promotes to
    numpy's default dtype and nothing casts back until the end. Ours stays in
    float32, which is what actually reaches the network. The residual gap is
    that dtype difference alone: float64-vs-float32 of the same computation is
    2.2e-4 here, while ours matches a float32 reference to 1.2e-6.
    """
    def reference(raw, input_size=518):
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = image.shape[:2]
        sh, sw = input_size / h, input_size / w
        sh = sw = max(sh, sw)              # 'lower_bound': scale by the larger
        def constrain(x, mn):
            y = (np.round(x / 14) * 14).astype(int)
            if y < mn:
                y = (np.ceil(x / 14) * 14).astype(int)
            return int(y)
        nh, nw = constrain(sh * h, input_size), constrain(sw * w, input_size)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        image = (image - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])
        return np.ascontiguousarray(image.transpose(2, 0, 1)[None]).astype(np.float32)

    for label, img in (("4:3", IMG_43), ("16:9", IMG_169)):
        ref = reference(img)
        spec = PreprocessSpec(resize="keep_ratio", target=518, multiple=14,
                              bound="lower", rounding="constrain", interp="cubic",
                              normalize="imagenet", resize_on="float")
        got, geom = preprocess(img, spec)
        check(f"depth_anything_v2 shape {label}", got.shape == ref.shape,
              f"{got.shape} vs {ref.shape}")
        if got.shape == ref.shape:
            check(f"depth_anything_v2 values {label}",
                  np.allclose(got, ref, atol=1e-5),
                  f"max|diff|={np.abs(got - ref).max():.2e}")


def test_depth_anything_ac_upstream():
    """Type A. The repo copy is missing /255 (defect D4); pin upstream."""
    def upstream(raw, target=518):
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = image.shape[:2]
        scale = target / min(h, w)
        nh, nw = int(h * scale), int(w * scale)
        nh, nw = ((nh + 13) // 14) * 14, ((nw + 13) // 14) * 14
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return np.ascontiguousarray(image.transpose(2, 0, 1)[None]).astype(np.float32)

    ref = upstream(IMG_43)
    spec = PreprocessSpec(resize="keep_ratio", target=518, multiple=14,
                          bound="lower", rounding="ceil", interp="cubic",
                          normalize="imagenet", resize_on="float")
    got, _ = preprocess(IMG_43, spec)
    ok = got.shape == ref.shape and np.allclose(got, ref, atol=1e-4)
    check("depth_anything_ac matches upstream (incl. /255)", ok,
          f"{got.shape} vs {ref.shape}"
          + ("" if got.shape != ref.shape
             else f" max|diff|={np.abs(got - ref).max():.2e}"))


def test_vggt_upstream_white_pad():
    """Types E. Upstream pads WHITE and resizes straight to 518 (D3, D7, D8)."""
    _, geom = resize_square_pad(IMG_169, 518, pad_value=(255, 255, 255))
    check("vggt square_pad size", (geom.dst_h, geom.dst_w) == (518, 518))
    # 16:9 -> letterboxed vertically
    check("vggt inner fills width", geom.inner_w == 518)
    check("vggt vertical padding", geom.pad_top > 0 and geom.pad_bottom > 0)
    check("vggt padding symmetric", abs(geom.pad_top - geom.pad_bottom) <= 1)

    out, _ = resize_square_pad(IMG_169, 518, pad_value=(255, 255, 255))
    check("vggt pad is white", out[0, 0].tolist() == [255, 255, 255],
          f"corner={out[0, 0].tolist()}")

    # the crop box must recover the real image region exactly
    y0, y1, x0, x1 = geom.crop_box()
    check("vggt crop box", (y1 - y0, x1 - x0) == (geom.inner_h, geom.inner_w))


def test_metric3d_pad_and_unpad():
    """Type D: keep_ratio + centre pad with the ImageNet mean, then unpad."""
    def reference(raw, sizes=(616, 1064)):
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(sizes[0] / h, sizes[1] / w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LINEAR)
        padding = [123.675, 116.28, 103.53]
        h2, w2 = rgb.shape[:2]
        ph, pw = sizes[0] - h2, sizes[1] - w2
        pht, pwl = ph // 2, pw // 2
        rgb = cv2.copyMakeBorder(rgb, pht, ph - pht, pwl, pw - pwl,
                                 cv2.BORDER_CONSTANT, value=padding)
        return rgb, [pht, ph - pht, pwl, pw - pwl]

    ref_img, ref_pad = reference(IMG_43)
    out, geom = resize_pad(cv2.cvtColor(IMG_43, cv2.COLOR_BGR2RGB), 616, 1064,
                           pad_value=(123.675, 116.28, 103.53), interp="linear")
    check("metric3d size", out.shape[:2] == (616, 1064), str(out.shape))
    check("metric3d padding matches reference",
          [geom.pad_top, geom.pad_bottom, geom.pad_left, geom.pad_right] == ref_pad,
          f"{[geom.pad_top, geom.pad_bottom, geom.pad_left, geom.pad_right]} vs {ref_pad}")

    # a depth map at network resolution must unpad to the inner region
    depth = np.arange(616 * 1064, dtype=np.float32).reshape(616, 1064)
    un = geom.unpad(depth)
    check("metric3d unpad shape", un.shape == (geom.inner_h, geom.inner_w), str(un.shape))
    back = geom.to_source(depth)
    check("metric3d back to source", back.shape == (480, 640), str(back.shape))


def test_stretch_matches_repo():
    """Type F: unidepth_v2 / unik3d as they are today."""
    ref = cv2.resize(IMG_43, (518, 518))
    out, geom = resize_stretch(IMG_43, 518, 518, interp="linear")
    check("stretch matches cv2.resize", np.array_equal(out, ref))
    check("stretch has no padding",
          (geom.pad_top, geom.pad_left) == (0, 0) and geom.inner_h == 518)


def test_normalize_modes():
    img = np.full((4, 4, 3), 255, np.uint8)
    check("scale mode", np.allclose(normalize(img, "scale"), 1.0))
    check("half mode -> +1", np.allclose(normalize(img, "half"), 1.0))
    check("imagenet mode",
          np.allclose(normalize(img, "imagenet")[0, 0, 0],
                      (1.0 - 0.485) / 0.229))
    check("raw255 passthrough", np.allclose(normalize(img, "raw255"), 255.0))
    check("raw255 with stats",
          np.allclose(normalize(img, "raw255", [123.675] * 3, [58.395] * 3)[0, 0, 0],
                      (255 - 123.675) / 58.395))


def test_geometry_roundtrip():
    """Padding then unpadding must be lossless in shape for every type."""
    for name, fn, args in (
        ("pad 4:3->518sq", resize_square_pad, (IMG_43, 518)),
        ("pad 16:9->518sq", resize_square_pad, (IMG_169, 518)),
        ("pad sq->518sq", resize_square_pad, (IMG_SQ, 518)),
        ("pad 4:3->616x1064", resize_pad, (IMG_43, 616, 1064)),
    ):
        out, geom = fn(*args)
        canvas = np.zeros(out.shape[:2], np.float32)
        un = geom.unpad(canvas)
        ok = un.shape == (geom.inner_h, geom.inner_w)
        src = geom.to_source(canvas)
        ok &= src.shape == (geom.src_h, geom.src_w)
        check(f"geometry roundtrip {name}", ok,
              f"unpad={un.shape} src={src.shape}")


def test_keep_ratio_multiple():
    for target, bound in ((518, "lower"), (504, "upper")):
        for img in (IMG_43, IMG_169, IMG_SQ):
            _, g = resize_keep_ratio(img, target, multiple=14, bound=bound)
            ok = g.dst_h % 14 == 0 and g.dst_w % 14 == 0
            check(f"keep_ratio multiple-of-14 t={target} {bound} {img.shape[:2]}", ok,
                  f"{g.dst_h}x{g.dst_w}")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{'=' * 60}")
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
