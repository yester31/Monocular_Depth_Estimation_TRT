"""core/preprocess.py must reproduce, byte for byte, what it replaced.

The acceptance test is `np.array_equal` against `reports/inputs/<model>.npy` --
the tensor each model's own onnx2trt.py handed its engine, and the tensor
verify_accuracy.py replays through the ONNX graph. `np.allclose` is not the
same test and is not accepted here: if this module produced a tensor that were
merely close, all fourteen accuracy baselines would shift by an unknown amount
and every later comparison would be against something nobody can name.

Those .npy files are read here and never written. Regenerating them would
destroy the only thing that makes this checkable without a GPU.

The other half is the call sites. `models/<model>/onnx2trt.py` cannot be
imported on a machine without TensorRT, so it is parsed instead: each converted
script must call preprocess_for() with its own name and the same input_h/input_w
that tests/test_spec.py already ties to spec.json, and must no longer carry its
own normalisation arithmetic. Same image plus same size plus same function is
then the whole argument.

Run:  python -m pytest tests/test_preprocess.py -q
      python tests/test_preprocess.py          (no pytest needed)
"""

import ast
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import preprocess as pp                                   # noqa: E402
from core import spec as spec_mod                                   # noqa: E402
from core.preprocess import (                                       # noqa: E402
    Geometry, PreprocessSpec, ResizeStep, normalize, preprocess, resize_keep_ratio,
    resize_pad, resize_snap, resize_square_pad, resize_stretch, scale,
    standardize, to_nchw,
)

INPUTS = os.path.join(ROOT, "reports", "inputs")
EXAMPLE = os.path.join(ROOT, "data", "example.jpg")
MODELS_DIR = os.path.join(ROOT, "models")

RNG = np.random.default_rng(0)
IMG_43 = RNG.integers(0, 256, (480, 640, 3), dtype=np.uint8)    # 4:3
IMG_169 = RNG.integers(0, 256, (720, 1280, 3), dtype=np.uint8)  # 16:9
IMG_SQ = RNG.integers(0, 256, (500, 500, 3), dtype=np.uint8)
SHAPES = (("4:3", IMG_43), ("16:9", IMG_169), ("square", IMG_SQ))

# Every model this module claims. depth_pro is deliberately absent -- see
# core.preprocess.UNSUPPORTED and docs/findings.md P1.
CONVERTED = sorted(pp.MODELS)


def _specs():
    return spec_mod.load_all()


def _example():
    img = cv2.imread(EXAMPLE)
    if img is None:
        pytest.skip(f"no {EXAMPLE}")
    return img


# ---------------------------------------------------------------------------
# The test that matters: the recorded tensor, byte for byte
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", CONVERTED)
def test_matches_the_recorded_input_tensor_exactly(model):
    """core/preprocess.py against reports/inputs/<model>.npy, np.array_equal."""
    oracle_path = os.path.join(INPUTS, f"{model}.npy")
    if not os.path.exists(oracle_path):
        pytest.skip(f"no reports/inputs/{model}.npy to check against")
    specs = _specs()
    if model not in specs:
        pytest.skip(f"no spec.json for {model}")

    oracle = np.load(oracle_path)
    got, geom = pp.preprocess_for(_example(), model, spec_mod.size_of(specs[model]))

    assert got.shape == oracle.shape, f"{got.shape} against recorded {oracle.shape}"
    assert got.dtype == oracle.dtype, f"{got.dtype} against recorded {oracle.dtype}"
    assert got.flags["C_CONTIGUOUS"], "the engine is handed a contiguous buffer"
    if not np.array_equal(got, oracle):
        d = float(np.abs(got.astype(np.float64) - oracle.astype(np.float64)).max())
        n = int((got != oracle).sum())
        raise AssertionError(
            f"{model}: {n} of {got.size} elements differ, max |diff| {d:.3e}. "
            f"This is not a rounding detail -- reports/inputs/{model}.npy is the "
            f"tensor the accuracy baseline was computed from, so a difference "
            f"here moves that baseline. Leave the model on its own path instead "
            f"(see PLAN.md section 5's stop condition).")
    assert (geom.src_h, geom.src_w) == _example().shape[:2]


def test_depth_pro_is_declined_rather_than_approximated():
    """The stop condition, as code: refuse instead of returning something close.

    cv2's INTER_LINEAR is the same mathematics as torch's bilinear here and
    still differs from the recorded tensor at 1.788e-07 on 34% of its elements,
    because the two accumulate the four taps in a different order.
    """
    assert "depth_pro" in pp.UNSUPPORTED
    assert "depth_pro" not in pp.MODELS
    with pytest.raises(pp.UnsupportedPreprocess):
        pp.spec_for("depth_pro", (1536, 1536))


# ---------------------------------------------------------------------------
# The call sites
# ---------------------------------------------------------------------------

def _call_site(model):
    """(model name argument, size argument source) of the preprocess_for call."""
    path = os.path.join(MODELS_DIR, model, "onnx2trt.py")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in ("preprocess_for", "spec_for"):
            continue
        args = node.args
        named = next((a.value for a in args
                      if isinstance(a, ast.Constant) and isinstance(a.value, str)), None)
        size = next((ast.unparse(a) for a in args if isinstance(a, ast.Tuple)), None)
        return named, size
    return None, None


@pytest.mark.parametrize("model", CONVERTED)
def test_the_script_actually_calls_the_shared_code(model):
    """A byte-exact module the build script does not call changes nothing."""
    path = os.path.join(MODELS_DIR, model, "onnx2trt.py")
    if not os.path.exists(path):
        pytest.skip(f"no models/{model}/onnx2trt.py")
    named, size = _call_site(model)
    assert named == model, (
        f"models/{model}/onnx2trt.py must call core.preprocess with its own "
        f"name; found {named!r}")
    assert size == "(input_h, input_w)", (
        f"models/{model}/onnx2trt.py passes {size!r}. It has to be the same "
        f"input_h/input_w tests/test_spec.py ties to spec.json, or the tensor "
        f"checked above is not the tensor the engine gets.")


@pytest.mark.parametrize("model", CONVERTED)
def test_the_script_kept_no_private_copy_of_the_arithmetic(model):
    """Two implementations of one thing is the state P1 exists to end."""
    path = os.path.join(MODELS_DIR, model, "onnx2trt.py")
    if not os.path.exists(path):
        pytest.skip(f"no models/{model}/onnx2trt.py")
    code = "\n".join(l for l in open(path, encoding="utf-8").read().splitlines()
                     if not l.lstrip().startswith("#"))
    tree = ast.parse(code)
    leftovers = [n.name for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)
                 and n.name in ("preprocess_image", "constrain_to_multiple_of")]
    assert not leftovers, (
        f"models/{model}/onnx2trt.py still defines {leftovers}; the shared path "
        f"and the private one will drift")


# ---------------------------------------------------------------------------
# The two identities the shared pipeline is built on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interp", ["linear", "cubic", "area"])
@pytest.mark.parametrize("dtype", ["uint8", "float32", "float64"])
def test_resizing_to_the_size_it_already_is_is_the_identity(interp, dtype):
    """Why the Depth-Anything family's second stage is free.

    main() squashes the frame to the engine's size, and preprocess_image()'s
    keep-ratio rule then resizes it to the size it already has. If that were not
    exactly the identity, modelling the two stages faithfully would still be
    wrong, because the recorded tensor went through both.
    """
    img = IMG_SQ.astype(np.dtype(dtype)) if dtype != "uint8" else IMG_SQ
    out = cv2.resize(img, (img.shape[1], img.shape[0]),
                     interpolation=pp.INTERP[interp])
    assert np.array_equal(out, img)


@pytest.mark.parametrize("interp", ["linear", "cubic", "area"])
def test_colour_swap_and_resize_commute(interp):
    """Why the shared pipeline may convert first though half the models do not.

    cv2.resize interpolates each channel independently, so permuting the
    channels before or after cannot change a value.
    """
    src = IMG_169
    a = cv2.cvtColor(cv2.resize(src, (518, 518), interpolation=pp.INTERP[interp]),
                     cv2.COLOR_BGR2RGB)
    b = cv2.resize(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), (518, 518),
                   interpolation=pp.INTERP[interp])
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Arithmetic dtype: the thing spec.json has no field for
# ---------------------------------------------------------------------------

def test_the_scale_dtype_is_not_cosmetic():
    """float64 then cast, against float32 throughout, on the same image.

    If these agreed there would be no reason for core/preprocess.py to carry
    dtypes at all. They do not, and six models declare the same `stretch` +
    `imagenet` pair while sitting on different sides of it.
    """
    f64 = normalize(IMG_43, "imagenet", scale_dtype="float64",
                    norm_dtype="float64").astype(np.float32)
    f32 = normalize(IMG_43, "imagenet", scale_dtype="float32",
                    norm_dtype="float32").astype(np.float32)
    assert not np.array_equal(f64, f32)
    assert np.allclose(f64, f32, atol=1e-5), "and yet allclose would pass either"


def test_scale_and_standardize_land_in_the_declared_dtype():
    assert scale(IMG_43, "imagenet", "float64").dtype == np.float64
    assert scale(IMG_43, "imagenet", "float32").dtype == np.float32
    # depth_anything_ac: float32 divide, float64 statistics
    x = scale(IMG_43, "imagenet", "float32")
    assert standardize(x, "imagenet", dtype="float64").dtype == np.float64
    assert standardize(x, "imagenet", dtype="float32").dtype == np.float32


def test_normalise_modes():
    img = np.full((4, 4, 3), 255, np.uint8)
    assert np.allclose(normalize(img, "scale"), 1.0)
    assert np.allclose(normalize(img, "over_255"), 1.0)      # spec.json's name
    assert np.allclose(normalize(img, "half"), 1.0)
    assert np.allclose(normalize(img, "imagenet")[0, 0, 0], (1.0 - 0.485) / 0.229)
    assert np.allclose(normalize(img, "none"), 255.0)
    assert np.allclose(normalize(img, "raw255"), 255.0)
    assert np.allclose(normalize(img, "raw255", [123.675] * 3, [58.395] * 3)[0, 0, 0],
                       (255 - 123.675) / 58.395)
    with pytest.raises(ValueError):
        normalize(img, "no-such-mode")


def test_rank_five_models_get_a_rank_five_tensor():
    blob = to_nchw(np.zeros((8, 8, 3), np.float32), rank=5)
    assert blob.shape == (1, 1, 3, 8, 8)
    assert to_nchw(np.zeros((8, 8, 3), np.float32)).shape == (1, 3, 8, 8)


# ---------------------------------------------------------------------------
# 4:3, 16:9 and square through every type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,img", SHAPES)
@pytest.mark.parametrize("model", CONVERTED)
def test_every_aspect_ratio_survives_every_model(model, label, img):
    """Shape, dtype, finiteness and the source size, for all three aspects.

    The recorded oracle is one 4:3 photograph. These three say the pipeline does
    not fall over on the other two, which is what the evaluation set will hand
    it the moment it stops being DIODE.
    """
    specs = _specs()
    if model not in specs:
        pytest.skip(f"no spec.json for {model}")
    h, w = spec_mod.size_of(specs[model])
    blob, geom = pp.preprocess_for(img, model, (h, w))
    rank = pp.spec_for(model, (h, w)).rank
    assert blob.shape == ((1, 1, 3, h, w) if rank == 5 else (1, 3, h, w)), label
    assert blob.dtype == np.float32
    assert np.isfinite(blob).all()
    assert (geom.src_h, geom.src_w) == img.shape[:2]
    assert (geom.dst_h, geom.dst_w) == (h, w)


@pytest.mark.parametrize("label,img", SHAPES)
def test_keep_ratio_preserves_the_aspect_ratio_it_promises(label, img):
    """depth_anything_ac's `native` profile is the one live user of this type."""
    blob, geom = pp.preprocess_for(img, "depth_anything_ac", (518, 518),
                                   stretch=False)
    assert geom.dst_h % 14 == 0 and geom.dst_w % 14 == 0, f"{geom.dst_h}x{geom.dst_w}"
    assert min(geom.dst_h, geom.dst_w) >= 518
    src_ratio = img.shape[1] / img.shape[0]
    out_ratio = geom.dst_w / geom.dst_h
    assert abs(out_ratio - src_ratio) < 0.06, f"{out_ratio:.3f} vs {src_ratio:.3f}"
    assert blob.shape == (1, 3, geom.dst_h, geom.dst_w)


def test_the_two_rounding_rules_really_disagree():
    """700 against 686 for a 4:3 frame at target 518 -- the reason `rounding`
    is a parameter rather than a choice made once."""
    _, ceil = resize_keep_ratio(IMG_43, 518, 14, "lower", "cubic", "ceil")
    _, constrain = resize_keep_ratio(IMG_43, 518, 14, "lower", "cubic", "constrain")
    assert (ceil.dst_h, ceil.dst_w) == (518, 700)
    assert (constrain.dst_h, constrain.dst_w) == (518, 686)


def test_snap_is_a_stretch_and_not_a_keep_ratio():
    """distill_any_depth's second stage, which upstream's name hides."""
    _, snap = resize_snap(IMG_169, 518, 518, multiple=14)
    assert (snap.dst_h, snap.dst_w) == (518, 518)
    _, keep = resize_keep_ratio(IMG_169, 518, 14, "lower", "cubic", "constrain")
    assert keep.dst_w != keep.dst_h


# ---------------------------------------------------------------------------
# Geometry: the coordinate contract
# ---------------------------------------------------------------------------

def test_metric3d_pad_matches_the_arithmetic_it_replaced():
    """The truncating resize and the centre pad, against the inline reference."""
    def reference(raw, sizes=(616, 1064)):
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = min(sizes[0] / h, sizes[1] / w)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        h2, w2 = rgb.shape[:2]
        ph, pw = sizes[0] - h2, sizes[1] - w2
        pht, pwl = ph // 2, pw // 2
        rgb = cv2.copyMakeBorder(rgb, pht, ph - pht, pwl, pw - pwl,
                                 cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
        return rgb, [pht, ph - pht, pwl, pw - pwl]

    # 400x411 is deliberate: it is a size where int(w * scale) is 632 and
    # round(w * scale) is 633, so it separates the truncating rule from the
    # rounding one. None of the three ordinary aspect ratios does -- which is
    # why leaving this to a "reasonable" round would have looked fine.
    odd_scale = RNG.integers(0, 256, (400, 411, 3), dtype=np.uint8)
    for label, img in list(SHAPES) + [("trunc != round", odd_scale)]:
        ref_img, ref_pad = reference(img)
        out, geom = resize_pad(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 616, 1064,
                               pad_value=(123.675, 116.28, 103.53), interp="linear",
                               rounding="trunc")
        assert np.array_equal(out, ref_img), label
        assert [geom.pad_top, geom.pad_bottom,
                geom.pad_left, geom.pad_right] == ref_pad, label

    _, trunc = resize_pad(odd_scale, 616, 1064, interp="linear", rounding="trunc")
    _, rnd = resize_pad(odd_scale, 616, 1064, interp="linear", rounding="round")
    assert (trunc.inner_w, rnd.inner_w) == (632, 633), (trunc.inner_w, rnd.inner_w)


def test_metric3d_unpad_inverts_the_pad():
    """The postprocess crops with exactly these four numbers, so a depth map at
    network resolution must come back on the source grid."""
    for label, img in SHAPES:
        _, geom = pp.preprocess_for(img, "metric3d_v2", (616, 1064))
        depth = np.arange(616 * 1064, dtype=np.float32).reshape(616, 1064)
        assert geom.unpad(depth).shape == (geom.inner_h, geom.inner_w), label
        assert geom.to_source(depth).shape == img.shape[:2], label


def test_vggt_coordinates_match_the_arithmetic_they_replaced():
    """Geometry.box must be the four floats the inline code produced.

    Including the quirk: the border is `top` on both sides rather than
    `max_dim - h - top`, so an odd difference leaves the square one pixel short.
    Reproduced, not corrected -- correcting it moves the tensor.
    """
    def reference(raw, input_h=518, input_w=518):
        height, width = raw.shape[:2]
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        max_dim = max(width, height)
        left, top = (max_dim - width) // 2, (max_dim - height) // 2
        s = input_w / max_dim
        img = cv2.copyMakeBorder(img, top, top, left, left, cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])
        img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
        blob = np.ascontiguousarray(
            img.transpose(2, 0, 1)[None][None].astype(np.float32) / np.float32(255.0))
        return blob, (left * s, top * s, (left + width) * s, (top + height) * s)

    odd = np.ascontiguousarray(IMG_43[:479])        # 479x640: odd difference
    for label, img in list(SHAPES) + [("odd 4:3", odd)]:
        ref_blob, ref_box = reference(img)
        got, geom = pp.preprocess_for(img, "vggt", (518, 518))
        assert np.array_equal(got, ref_blob), label
        assert geom.box == ref_box, f"{label}: {geom.box} vs {ref_box}"
        # and the integer crop the postprocess actually slices with
        y0, y1, x0, x1 = geom.crop_box()
        assert (y0, y1, x0, x1) == (int(round(ref_box[1])), int(round(ref_box[3])),
                                    int(round(ref_box[0])), int(round(ref_box[2]))), label


def test_vggt_pads_white_not_black():
    out, geom = resize_square_pad(IMG_169, 518, 518, pad_value=(255, 255, 255))
    assert out[0, 0].tolist() == [255, 255, 255]
    assert geom.pad_top > 0 and geom.pad_bottom > 0
    assert geom.pad_left == 0 and geom.pad_right == 0


def test_stretch_records_no_padding():
    for label, img in SHAPES:
        out, geom = resize_stretch(img, 518, 518, interp="linear")
        assert np.array_equal(out, cv2.resize(img, (518, 518))), label
        assert (geom.pad_top, geom.pad_bottom, geom.pad_left, geom.pad_right) == (0,) * 4
        assert (geom.inner_h, geom.inner_w) == (518, 518)
        assert geom.to_source(np.zeros((518, 518), np.float32)).shape == img.shape[:2]


@pytest.mark.parametrize("model", CONVERTED)
@pytest.mark.parametrize("label,img", SHAPES)
def test_the_geometry_round_trips_for_every_model(model, label, img):
    """Unpad then resize must land back on the source grid, whatever the type."""
    specs = _specs()
    if model not in specs:
        pytest.skip(f"no spec.json for {model}")
    h, w = spec_mod.size_of(specs[model])
    _, geom = pp.preprocess_for(img, model, (h, w))
    canvas = np.zeros((h, w), np.float32)
    assert geom.unpad(canvas).shape == (geom.inner_h, geom.inner_w)
    assert geom.to_source(canvas).shape == img.shape[:2]


# ---------------------------------------------------------------------------
# The declaration and the table must not drift apart
# ---------------------------------------------------------------------------

# spec.json calls both `pad` and `square_pad` "keep_ratio_pad". Written down
# rather than silently accepted: the two are different functions with different
# coordinate bookkeeping, and P1 does not edit spec.json.
_DECLARED_TYPE = {"stretch": {"stretch", "none"},
                  "keep_ratio_pad": {"pad", "square_pad"}}


@pytest.mark.parametrize("model", CONVERTED)
def test_the_table_agrees_with_spec_json(model):
    specs = _specs()
    if model not in specs:
        pytest.skip(f"no spec.json for {model}")
    declared = specs[model]["preprocess"]
    s = pp.spec_for(model, spec_mod.size_of(specs[model]))

    assert s.resize.kind in _DECLARED_TYPE[declared["type"]], (
        f"{model}: spec.json says {declared['type']}, table builds "
        f"{s.resize.kind}")
    assert pp.NORMALISE_ALIASES.get(declared["normalise"], declared["normalise"]) \
        == pp.NORMALISE_ALIASES.get(s.normalize, s.normalize), (
        f"{model}: spec.json says {declared['normalise']}, table says {s.normalize}")
    if declared["interpolation"] in pp.INTERP:
        assert s.resize.interp == declared["interpolation"], (
            f"{model}: spec.json says {declared['interpolation']}, table says "
            f"{s.resize.interp}")


def test_depth_pro_is_the_only_model_without_an_entry():
    specs = set(_specs())
    assert specs - set(pp.MODELS) - set(pp.UNSUPPORTED) == set()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
