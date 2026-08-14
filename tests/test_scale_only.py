"""Scale-only alignment, checked against a scene whose true scale is known.

VGGT and StreamVGGT return a normalised coordinate frame: the point map was
divided by the average Euclidean distance of its points to the origin, so the
depth that comes out is in units of one average scene distance rather than
metres. There is no engine and no DIODE on the machine this was written on, so
the check is built the other way round -- a point cloud is constructed from a
depth map whose metres are known, normalised exactly the way the paper says,
and the alignment then has to hand the original scale back.

That is the whole claim being made about these two models: **one number is
missing and it is recoverable.** If the recovery were wrong, or if it quietly
became a two-parameter fit, these tests fail before any GPU time is spent.

The metric path is not touched here. `align` and `metrics` carry eleven
published scores and tests/test_gt.py pins them; this file only exercises what
was added beside them.
"""

import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gt import (align, metrics, policy_for, policy_for_normalised,  # noqa: E402
                     scale_only)


# --------------------------------------------------------------------------
# A synthetic scene, and the normalisation VGGT applies to its training data.
# --------------------------------------------------------------------------

def synthetic_depth(h=48, w=64, fx=70.0, fy=70.0):
    """A depth map in metres with real structure in it.

    A tilted floor with a box standing on it. Structure matters: a constant
    depth map would let any fit look perfect, and a plane alone would hide a
    mistake in how the point map is built.
    """
    v, u = np.meshgrid(np.arange(h, dtype=np.float64),
                       np.arange(w, dtype=np.float64), indexing="ij")
    z = 2.0 + 0.05 * v + 0.01 * u            # floor receding from the camera
    z[10:30, 15:40] = 1.4                    # a box in front of it
    return z, (fx, fy, w / 2.0, h / 2.0)


def point_map(z, intr):
    """Back-project depth to camera-space points, the map VGGT predicts."""
    fx, fy, cx, cy = intr
    h, w = z.shape
    v, u = np.meshgrid(np.arange(h, dtype=np.float64),
                       np.arange(w, dtype=np.float64), indexing="ij")
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)


def normalise(points):
    """VGGT's normalisation: divide by the mean distance of the points to the
    origin. arXiv:2503.11651 -- "we compute the average Euclidean distance of
    all 3D points in the point map P to the origin and use this scale to
    normalize the camera translations t, the point map P, and the depth map D".

    Returns (normalised points, the constant that was divided out). That
    constant is the answer every recovery test below is checked against.
    """
    c = float(np.mean(np.sqrt((points ** 2).sum(axis=-1))))
    return points / c, c


def test_the_fit_returns_the_constant_that_was_divided_out():
    z, intr = synthetic_depth()
    normalised, c = normalise(point_map(z, intr))
    pred = normalised[..., 2]                # the depth channel, in its own unit
    m = np.ones_like(z, dtype=bool)

    out, fit = scale_only(pred, z, m)

    # Exact, not approximate: pred is c times g everywhere, so the least
    # squares problem has a zero residual and one solution.
    assert fit["scale"] == pytest.approx(c, rel=1e-12)
    assert out == pytest.approx(z, rel=1e-12)
    assert c > 1.0                           # the scene really was rescaled


def test_the_recovered_scene_scores_as_perfectly_as_a_metric_model_could():
    z, intr = synthetic_depth()
    normalised, _ = normalise(point_map(z, intr))
    m = np.ones_like(z, dtype=bool)
    aligned, _ = scale_only(normalised[..., 2], z, m)
    r = metrics(aligned, z, m)
    assert r["abs_rel"] < 1e-12
    assert r["delta1"] == pytest.approx(1.0)


def test_the_same_prediction_unaligned_is_not_a_depth_in_metres():
    """Why these rows cannot go in the metric table.

    The identical prediction that scores perfectly above scores like a broken
    model when it is read as metres, because it is not metres. The gap is the
    normalising constant and nothing else -- which is exactly why publishing
    the two in one ranked column would be publishing a unit as a quality.
    """
    z, intr = synthetic_depth()
    normalised, c = normalise(point_map(z, intr))
    m = np.ones_like(z, dtype=bool)
    raw = metrics(normalised[..., 2], z, m)
    assert raw["abs_rel"] > 0.5
    assert raw["delta1"] < 0.01
    # And the error is the scale: 1 - 1/c, to the last digit.
    assert raw["abs_rel"] == pytest.approx(1.0 - 1.0 / c, rel=1e-9)


def test_scale_only_fits_one_parameter_and_not_two():
    """The guard against this quietly becoming scale+shift.

    A prediction with a genuine additive offset cannot be repaired by a scale,
    and must not be. If someone ever routes this through an affine fit, the
    residual below goes to zero and this test says so.
    """
    z, _ = synthetic_depth()
    pred = (z + 0.8) / 3.0                   # offset as well as scale
    m = np.ones_like(z, dtype=bool)
    aligned, fit = scale_only(pred, z, m)
    assert set(fit) == {"scale"}
    assert metrics(aligned, z, m)["abs_rel"] > 0.02


def test_the_fitted_scale_is_the_least_squares_minimiser():
    """<p,g>/<p,p> is checked against a scan, not against itself.

    A closed form is only worth having if it is the right closed form, and the
    reason it is written by hand rather than handed to np.linalg.lstsq is in
    core/gt.py: LAPACK aborts this process once torch has been imported. So
    the check is a scan over candidate scales, which needs no backend either.
    """
    z, intr = synthetic_depth()
    pred = normalise(point_map(z, intr))[0][..., 2]
    rng = np.random.default_rng(0)
    pred = pred * rng.uniform(0.9, 1.1, size=pred.shape)     # no exact answer
    m = np.ones_like(z, dtype=bool)
    _, fit = scale_only(pred, z, m)
    s = fit["scale"]

    def sse(k):
        return float(np.sum((k * pred - z) ** 2))

    best = sse(s)
    for k in np.linspace(s * 0.9, s * 1.1, 401):
        assert sse(float(k)) >= best - 1e-9


def test_noise_moves_the_score_but_barely_moves_the_scale():
    """One parameter fitted over thousands of pixels is a stable thing.

    This is what makes the fitted scale worth publishing as a diagnostic: if it
    wanders between images, that is the model's inconsistency showing, not the
    fit's.
    """
    z, intr = synthetic_depth()
    truth = normalise(point_map(z, intr))
    pred, c = truth[0][..., 2], truth[1]
    rng = np.random.default_rng(7)
    noisy = pred * rng.normal(1.0, 0.05, size=pred.shape)
    m = np.ones_like(z, dtype=bool)
    _, fit = scale_only(noisy, z, m)
    assert fit["scale"] == pytest.approx(c, rel=0.01)
    assert metrics(*(scale_only(noisy, z, m)[0], z, m))["abs_rel"] > 0.01


def test_pixels_outside_the_mask_do_not_reach_the_fit():
    """The scanner's mask is the only thing that decides what is fitted.

    DIODE leaves large regions unmeasured, and a normalised model will happily
    predict something for them. Letting those pixels into the fit would set the
    unit from a region with no measurement behind it.
    """
    z, intr = synthetic_depth()
    pred = normalise(point_map(z, intr))[0][..., 2]
    c = normalise(point_map(z, intr))[1]
    corrupted = pred.copy()
    corrupted[:12, :] = 1e3                  # nonsense where the scanner failed
    m = np.ones_like(z, dtype=bool)
    m[:12, :] = False
    _, fit = scale_only(corrupted, z, m)
    assert fit["scale"] == pytest.approx(c, rel=1e-12)


def test_scale_only_is_align_scale_and_the_metric_paths_are_untouched():
    """The new entry point is a name, not a second implementation.

    Two implementations of one fit is how the published numbers and the new
    ones drift apart without anyone noticing.
    """
    z, _ = synthetic_depth()
    pred = z / 4.0
    m = np.ones_like(z, dtype=bool)
    a1, f1 = scale_only(pred, z, m)
    a2, f2 = align(pred, z, m, "scale")
    assert f1 == f2
    assert a1 == pytest.approx(a2)
    # And nothing has been added to what a declared contract earns.
    assert align(pred, z, m, "none")[1] == {}


# --------------------------------------------------------------------------
# Which models are allowed through this path
# --------------------------------------------------------------------------

def test_an_undeclared_contract_still_earns_nothing_by_itself():
    # The thing core/gt.py's docstring refuses to do: `unknown` must not become
    # an alignment because an alignment would produce a number.
    assert policy_for("unknown") is None
    assert policy_for_normalised("unknown") is None
    assert policy_for_normalised("metric") is None
    assert policy_for_normalised("relative") is None
    # A spec.json that declares the frame is a different matter.
    assert policy_for_normalised("normalised") == "scale"
    # The declared contracts keep the alignments the published rows used.
    assert policy_for("metric") == "none"
    assert policy_for("relative") == "scale_shift"


def test_the_scale_only_path_has_to_be_asked_for_by_name():
    import evaluate_gt

    unknown = {"depth_scale": "unknown"}
    # Without the flag, vggt answers exactly what it answered before the
    # adapter existed.
    policy, why = evaluate_gt.alignment_policy("vggt", unknown, scale_only=False)
    assert policy is None and "output contract" in why
    # With it, and only for the models whose normalisation is written down.
    assert evaluate_gt.alignment_policy("vggt", unknown, scale_only=True)[0] == "scale"
    assert evaluate_gt.alignment_policy("streamvggt", unknown,
                                        scale_only=True)[0] == "scale"
    policy, why = evaluate_gt.alignment_policy("depth_anything_v3", unknown,
                                               scale_only=True)
    assert policy is None and "normalised coordinate frame" in why


def test_a_declared_contract_outranks_the_flag():
    """--scale-only cannot be used to re-score a model that claims metres.

    Every metric row in reports/gt.md would look better under a fitted scale.
    The flag must not be a way to get that number into a table.
    """
    import evaluate_gt

    policy, why = evaluate_gt.alignment_policy(
        "vggt", {"depth_scale": "metric"}, scale_only=True)
    assert policy is None
    assert "not making" in why
    policy, why = evaluate_gt.alignment_policy(
        "vggt", {"depth_scale": "relative"}, scale_only=True)
    assert policy is None


def test_every_model_offered_the_flag_carries_its_reason():
    import evaluate_gt

    for model, why in evaluate_gt.NORMALISED_COORDINATES.items():
        assert model in evaluate_gt.ADAPTERS
        # A citation, not an assertion. The reason travels into the report.
        assert "arXiv" in why


# --------------------------------------------------------------------------
# The adapters themselves. These run without a GPU and without torch.
# --------------------------------------------------------------------------

def test_the_adapters_reproduce_the_tensor_the_benchmark_fed():
    """The oracle check, run here so it is not only available behind --check.

    reports/inputs/<model>.npy is the tensor that model's own onnx2trt.py put
    into the engine when it was benchmarked. An adapter that cannot reproduce
    it is preprocessing differently from the thing being scored.
    """
    import cv2

    import evaluate_gt

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example = cv2.imread(os.path.join(root, "data", "example.jpg"))
    assert example is not None
    for model in ("vggt", "streamvggt"):
        passed, detail = evaluate_gt.check_adapter(model, example, (518, 518))
        assert passed, f"{model}: {detail}"


def test_the_padding_is_white():
    """Black padding is a different input to the network.

    Upstream pads with 1.0 on a 0-1 tensor and both onnx2trt.py files were
    corrected to match. A regression here changes what the model sees wherever
    the source is not square, and nothing downstream would look broken.
    """
    import evaluate_gt

    bgr = np.zeros((40, 80, 3), dtype=np.uint8)          # a black landscape
    x = evaluate_gt._square_pad_pre(bgr, 64, 64)
    assert x.shape == (1, 1, 3, 64, 64)
    assert x[0, 0, :, 0, 0] == pytest.approx(1.0)        # padding, top-left
    assert x[0, 0, :, 32, 32] == pytest.approx(0.0)      # image, centre


def test_the_crop_puts_the_prediction_back_on_the_original_grid():
    """The letterbox comes off before anything is compared to a measurement.

    The padding is white paper the network invented a depth for; scoring it
    against a scanner reading would be scoring the pad. A smooth ramp goes
    through pad, resize, crop and resize back, and has to survive.
    """
    import cv2

    import evaluate_gt

    oh, ow, net = 96, 128, 64
    v, u = np.meshgrid(np.arange(oh, dtype=np.float64),
                       np.arange(ow, dtype=np.float64), indexing="ij")
    depth = 1.0 + 0.01 * u + 0.02 * v
    max_dim, pad_marker = max(oh, ow), 999.0
    top, left = (max_dim - oh) // 2, (max_dim - ow) // 2
    padded = cv2.copyMakeBorder(depth, top, top, left, left,
                                cv2.BORDER_CONSTANT, value=pad_marker)
    engine_out = cv2.resize(padded, (net, net), interpolation=cv2.INTER_LINEAR)

    got = evaluate_gt._square_pad_depth([engine_out.ravel()], (net, net), oh, ow)

    assert got.shape == (oh, ow)
    # Nothing from the padded region survived: it would be off by two orders
    # of magnitude if the crop were skipped or misplaced.
    assert float(np.nanmax(got)) < 10.0
    assert np.abs(got - depth).max() < 0.05


def test_one_non_finite_pixel_would_poison_the_whole_fit():
    """Why score_model narrows the mask before the scale fit.

    align("scale") is two dot products over the masked pixels and filters
    nothing itself -- that is fine for the eleven published scores, which never
    reach it, and not fine here: the vggt adapter marks a non-positive
    prediction as NaN rather than clipping it to a plausible number. One such
    pixel inside the scanner's mask turns the fitted scale into NaN and takes
    every metric with it, silently.
    """
    z, _ = synthetic_depth()
    pred = z / 2.0
    pred[5, 5] = np.nan
    m = np.ones_like(z, dtype=bool)

    poisoned = scale_only(pred, z, m)[1]["scale"]
    assert poisoned != poisoned                      # NaN, and nothing says so

    # What score_model actually does.
    ok = scale_only(pred, z, m & np.isfinite(pred))[1]["scale"]
    assert ok == pytest.approx(2.0)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(ROOT_DIR, "reports", "gt")
GT_TABLES = ("", "_fp32", "_uint8")


def _published(suffix):
    """(path, text) of one published table, or skip."""
    p = os.path.join(ROOT_DIR, "reports", f"gt{suffix}.md")
    if not (os.path.isdir(GT_DIR) and os.path.exists(p)):
        pytest.skip(f"reports/gt{suffix}.md and its JSON artefacts are not here")
    with open(p, encoding="utf-8") as f:
        return p, f.read()


def _table_rows(text):
    """model -> the cells of its row in the first table, by header name."""
    lines = [l for l in text.splitlines() if l.startswith("|")]
    if not lines:
        return {}
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = {}
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        m = re.match(r"`([a-z0-9_]+)`", cells[0]) if cells else None
        if not m or len(cells) != len(head):
            continue
        rows.setdefault(m.group(1), dict(zip(head, cells)))
    return rows


@pytest.mark.parametrize("suffix", GT_TABLES)
def test_the_published_table_still_renders_byte_for_byte(suffix):
    """The strongest statement available that the metric path did not move.

    Rendering the published table again from the JSON artefacts its scores were
    written from has to produce the same file -- same columns, same order, same
    closing paragraph.

    What this does *not* catch, and the reason the three tests below exist: it
    compares the generator against the file, so a change to both at once passes
    it. Adding the input-size and precision columns changed `render()` and
    `reports/gt.md` together and this test went green without being touched.
    It is a consistency check -- nobody hand-edited the published table, and
    nobody changed the renderer without republishing -- not a content check.
    """
    import json

    import evaluate_gt

    path, text = _published(suffix)
    others = [s for s in GT_TABLES if s and s != suffix]
    results = []
    for name in sorted(os.listdir(GT_DIR)):
        if not name.endswith(".json"):
            continue
        stem = name[:-5]
        if suffix:
            if not stem.endswith(suffix):
                continue
        elif stem.endswith(tuple(others)):
            continue
        with open(os.path.join(GT_DIR, name), encoding="utf-8") as f:
            results.append(json.load(f))
    # Only the rows the file publishes. reports/gt/ accumulates the artefacts of
    # every run, and there are eight *_fp32.json against the three rows
    # gt_fp32.md publishes. The exception is a not-measured record, which is
    # never a row and must still reach the renderer -- filtering by the row set
    # alone would delete exactly the records that exist to be un-deletable.
    from core import unmeasured

    wanted = set(_table_rows(text))
    results = [r for r in results
               if r["model"] in wanted or unmeasured.is_unmeasured(r)]
    # Reconstructed from the header line of the published file itself.
    manifest = {"source": {"dataset": "DIODE"}, "split": "indoors", "count": 50,
                "depth_units": "metres", "depth_convention": "z-depth"}
    assert evaluate_gt.render(results, manifest, False, suffix) == text


@pytest.mark.parametrize("suffix", GT_TABLES)
def test_the_published_table_states_its_conditions(suffix):
    """Execution rule 8, as a test on the artefact rather than on the renderer.

    Three files with identical headings published three different sets of
    conditions -- fp16, fp32, and a uint8-input engine -- and none of them said
    so. A reader comparing `reports/gt.md` with `reports/gt_fp32.md` had nothing
    in either table to tell them apart.
    """
    _, text = _published(suffix)
    rows = _table_rows(text)
    assert rows, "no model rows parsed out of the published table"
    for model, cells in rows.items():
        assert "input HxW" in cells, f"{model}: no input size column"
        assert re.fullmatch(r"\d+x\d+", cells["input HxW"]), \
            f"{model}: input size is {cells['input HxW']!r}"
        assert cells.get("precision") in ("fp16", "fp32"), \
            f"{model}: precision is {cells.get('precision')!r}"


@pytest.mark.parametrize("suffix", GT_TABLES)
def test_the_conditions_come_from_the_json_and_not_from_a_keyboard(suffix):
    """Execution rule 7. Every condition cell must be in the result file.

    The one that would be easy to get wrong is the input size: the GT result
    JSON does not carry input_h/input_w, so evaluate_gt.conditions() falls back
    to the NNNxNNN in the engine filename. This asserts that fallback against
    what the model's spec.json declares, which is an independent source.
    """
    import json

    import evaluate_gt
    from core import spec as spec_mod

    _, text = _published(suffix)
    specs = spec_mod.load_all()
    for model, cells in _table_rows(text).items():
        with open(os.path.join(GT_DIR, f"{model}{suffix}.json"), encoding="utf-8") as f:
            r = json.load(f)
        h, w, precision, dtype = evaluate_gt.conditions(r)
        assert cells["input HxW"] == f"{h}x{w}"
        assert cells["precision"] == precision
        assert model in specs, f"{model} has no spec.json to cross-check against"
        assert (h, w) == tuple(spec_mod.size_of(specs[model])), (
            f"{model}: the table says {h}x{w}, spec.json says "
            f"{spec_mod.size_of(specs[model])}")
        assert dtype == ("uint8" if suffix == "_uint8" else "float32"), \
            f"{model}: input dtype read as {dtype} in gt{suffix}.md"


@pytest.mark.parametrize("suffix", GT_TABLES)
def test_the_scores_in_the_table_are_the_scores_in_the_json(suffix):
    """The columns were added; the numbers must not have moved.

    Read straight off the published markdown and compared with the JSON, so
    this holds whatever the renderer does. If a future column shifts a value
    into the wrong cell, this is what says so.
    """
    import json

    _, text = _published(suffix)
    for model, cells in _table_rows(text).items():
        with open(os.path.join(GT_DIR, f"{model}{suffix}.json"), encoding="utf-8") as f:
            r = json.load(f)
        assert cells["AbsRel"] == f"{r['abs_rel'] * 100:.2f}%", model
        assert cells["RMSE"] == f"{r['rmse']:.3f}", model
        assert cells["log10"] == f"{r['log10']:.4f}", model
        for i in (1, 2, 3):
            assert cells[f"d{i}"] == f"{r[f'delta{i}'] * 100:.1f}%", f"{model} d{i}"
        assert cells["images"] == str(r["images"]), model
        assert cells["alignment"] == r["alignment"], model


def test_the_scale_only_report_is_a_different_table_and_says_so():
    import evaluate_gt

    manifest = {"source": {"dataset": "DIODE"}, "split": "indoors", "count": 50,
                "depth_units": "metres", "depth_convention": "z-depth"}
    results = [{"model": "vggt", "alignment": "scale", "abs_rel": 0.11,
                "rmse": 0.6, "log10": 0.05, "delta1": 0.88, "delta2": 0.97,
                "delta3": 0.99, "images": 50, "fit_median": {"scale": 2.34},
                "normalised_because": evaluate_gt.NORMALISED_COORDINATES["vggt"]}]
    out = evaluate_gt.render(results, manifest, scale_only=True)
    assert "fitted scale" in out and "2.340" in out
    assert "Do not merge it with" in out
    assert "arXiv:2503.11651" in out          # the reason travels with the number
    assert "one dot product over another" in out

    # And none of that reaches the metric table.
    plain = evaluate_gt.render(
        [dict(r, alignment="none") for r in results], manifest)
    assert "fitted scale" not in plain
    assert "Do not merge" not in plain


def test_the_crop_geometry_matches_the_models_own_arithmetic():
    """Same formula as models/*/onnx2trt.py, which tests/test_vggt_geometry.py
    pins independently. A landscape source loses top and bottom and keeps its
    full width."""
    import evaluate_gt

    x1, y1, x2, y2 = evaluate_gt._square_pad_geometry(720, 1280, 518)
    assert (x1, x2) == pytest.approx((0.0, 518.0))
    assert y1 > 1.0 and y2 < 517.0
    assert (y2 - y1) == pytest.approx(720 / 1280 * 518)
    # A square source loses nothing.
    assert evaluate_gt._square_pad_geometry(500, 500, 518) == pytest.approx(
        (0.0, 0.0, 518.0, 518.0))
