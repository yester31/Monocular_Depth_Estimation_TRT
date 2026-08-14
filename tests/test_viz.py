"""The demo figure has to be checkable without a GPU, and it has to not lie.

Two different things are tested here and they are not equally important.

The layout tests are ordinary: a figure gets written, three aspect ratios come
out the far end, a caption does not run into the next model's picture.

The one that matters is `test_a_metric_disagreement_survives_the_drawing`.
Colouring each result by its own minimum and maximum is the natural way to
write this demo and it silently deletes the only thing reports/gt.md ranks the
metric models on. That test states the property in the form of two models that
disagree by two metres, and fails if the drawing makes them look alike.

Nothing here needs TensorRT, an engine, or DIODE. The three aspect ratios are
synthesised in a temporary directory rather than read from data/eval, because
that manifest belongs to another task and this file has to run either way.
"""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

cv2 = pytest.importorskip("cv2")
pytest.importorskip("matplotlib")

from core import preprocess as pre_mod   # noqa: E402
from core import viz                     # noqa: E402

import demo                              # noqa: E402


# Height, width. Three aspect ratios, which is what P7 asks to be checked.
ASPECTS = {"4x3": (768, 1024), "16x9": (576, 1024), "portrait": (768, 576)}


def _image(h, w, seed=0):
    """A deterministic picture with structure, so a transpose is visible."""
    ys, xs = np.mgrid[0:h, 0:w]
    r = (xs * 255 // max(w - 1, 1)).astype(np.uint8)
    g = (ys * 255 // max(h - 1, 1)).astype(np.uint8)
    b = np.full((h, w), 40 + seed * 7, np.uint8)
    return np.stack([b, g, r], axis=-1)      # BGR, as cv2.imread returns


def _card(model="m", band="metric", h=518, w=518):
    return {"model": model, "input_h": h, "input_w": w, "band": band,
            "depth_scale": band, "unit": "m" if band == "metric" else "-",
            "precision": "fp16", "profile": "bench",
            "outputs": [("output", "metric depth")],
            "engine_fingerprint": "0123456789ab",
            "engine_fingerprint_source": "record", "engine_path": "",
            "engine_present": False, "mean_ms": 4.31, "device": "GPU",
            "preprocess": {"type": "stretch"}, "caveats": []}


# ---------------------------------------------------------------------------
# The property the whole module exists for
# ---------------------------------------------------------------------------

def test_metric_models_share_one_range_and_relative_models_do_not():
    cards = {"a": _card("a", "metric"), "b": _card("b", "metric"),
             "r": _card("r", "relative"), "u": _card("u", "unknown")}
    depths = {"a": np.full((8, 8), 2.0), "b": np.full((8, 8), 6.0),
              "r": np.linspace(0, 1, 64).reshape(8, 8),
              "u": np.linspace(0, 3, 64).reshape(8, 8)}
    ranges, note = viz.display_ranges(cards, depths)

    assert ranges["a"][:3] == ranges["b"][:3], (
        "two metric models were given different colour ranges; the figure "
        "would show their disagreement as agreement")
    assert ranges["r"][:2] != ranges["a"][:2]
    assert ranges["r"][2] != ranges["a"][2], (
        "a relative panel is drawn in the metric colour map, so a reader can "
        "put it on the metric colourbar")
    assert "shared metric range" in note
    for m in ("r", "u"):
        assert "not metres" in ranges[m][3]


def test_a_metric_disagreement_survives_the_drawing():
    """Two metric models 2 m apart must not produce the same picture."""
    shape = (32, 48)
    near = np.linspace(1.0, 3.0, shape[0] * shape[1]).reshape(shape)
    far = near + 2.0
    cards = {"near": _card("near"), "far": _card("far")}
    ranges, _ = viz.display_ranges(cards, {"near": near, "far": far})

    honest = [viz.colorize(d, *ranges[m][:3])
              for m, d in (("near", near), ("far", far))]
    gap = float(np.abs(honest[0].astype(float) - honest[1].astype(float)).mean())
    assert gap > 20.0, (
        f"on the shared range the two models differ by only {gap:.1f} of 255 "
        f"per channel; the metric difference is not visible")

    # The mistake this guards against, stated so the test explains itself: on
    # per-image min/max the two are the same picture up to rounding, and the
    # demo would be saying two models that disagree by 2 m agree.
    naive = [viz.colorize(d, *viz.robust_range([d]), viz.METRIC_CMAP)
             for d in (near, far)]
    naive_gap = float(np.abs(naive[0].astype(float)
                             - naive[1].astype(float)).mean())
    assert naive_gap < 1.0, (
        f"per-image normalisation no longer collapses the two ({naive_gap:.2f} "
        f"of 255) -- update the reasoning here rather than deleting the test")
    assert gap > 20 * naive_gap + 20


def test_fit_scale_shift_recovers_a_known_transform():
    rng = np.random.default_rng(3)
    pred = rng.uniform(0.2, 4.0, (40, 60))
    ref = 3.25 * pred - 0.75
    s, t = viz.fit_scale_shift(pred, ref)
    assert s == pytest.approx(3.25, rel=1e-6)
    assert t == pytest.approx(-0.75, abs=1e-6)


def test_fit_scale_shift_refuses_when_there_is_nothing_to_fit():
    a = np.full((4, 4), np.nan)
    with pytest.raises(ValueError):
        viz.fit_scale_shift(a, a)


def test_robust_range_ignores_nan_and_non_positive():
    a = np.array([np.nan, -5.0, 0.0, 1.0, 2.0, 3.0, 100.0])
    lo, hi = viz.robust_range([a], 0, 100)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Geometry: original size restored, padding drawn where it really is
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ASPECTS))
def test_stretch_geometry_restores_the_source_size(name):
    h, w = ASPECTS[name]
    spec = {"preprocess": {"type": "stretch"}}
    g = viz.input_geometry(spec, h, w, 518, 518)
    assert g.crop_box() == (0, 518, 0, 518)
    back = g.to_source(np.zeros((518, 518), np.float32))
    assert back.shape == (h, w)


@pytest.mark.parametrize("name", sorted(ASPECTS))
def test_pad_geometry_restores_the_source_size_and_stays_in_bounds(name):
    h, w = ASPECTS[name]
    spec = {"preprocess": {"type": "keep_ratio_pad"}}
    g = viz.input_geometry(spec, h, w, 616, 1064)
    y0, y1, x0, x1 = g.crop_box()
    assert 0 <= y0 < y1 <= 616 and 0 <= x0 < x1 <= 1064
    assert g.pad_top + g.inner_h + g.pad_bottom == 616
    assert g.pad_left + g.inner_w + g.pad_right == 1064
    back = g.to_source(np.zeros((616, 1064), np.float32))
    assert back.shape == (h, w)


@pytest.mark.parametrize("name", sorted(ASPECTS))
def test_pad_geometry_agrees_with_the_adapter_that_scores_metric3d(name):
    """The pad drawn on the figure has to be the pad evaluate_gt undoes.

    Two independent implementations of the same centred keep-ratio pad is
    exactly the kind of pair that drifts by one pixel and puts the dashed box
    somewhere the image is not.
    """
    try:
        import evaluate_gt
    except Exception as e:                                   # pragma: no cover
        pytest.skip(f"evaluate_gt not importable here: {type(e).__name__}: {e}")
    h, w = ASPECTS[name]
    size = evaluate_gt.METRIC3D_SIZE
    _, (rh, rw), pad = evaluate_gt._metric3d_geometry(h, w, size)
    g = viz.input_geometry({"preprocess": {"type": "keep_ratio_pad"}},
                           h, w, size[0], size[1])
    assert (g.inner_h, g.inner_w) == (rh, rw)
    assert (g.pad_top, g.pad_bottom, g.pad_left, g.pad_right) == tuple(pad)


def test_an_undeclared_resize_type_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        viz.input_geometry({"preprocess": {"type": "square_pad"}}, 10, 10, 20, 20)


def test_geometry_note_states_sizes_the_same_way_round_as_the_captions():
    g = viz.input_geometry({"preprocess": {"type": "keep_ratio_pad"}},
                           576, 1024, 518, 518)
    note = viz.geometry_note(g)
    assert note.startswith("hxw:")
    assert "pad t" in note and "l0 r0" in note


# ---------------------------------------------------------------------------
# What has to appear next to every panel
# ---------------------------------------------------------------------------

def test_the_card_carries_every_field_p7_requires():
    spec = {"name": "x", "depth_scale": "metric",
            "profiles": {"bench": {"size": [388, 518]}},
            "build_targets": [{"profile": "bench", "precision": "fp16"}],
            "outputs": [{"name": "points", "meaning": "point map"}]}
    run = {"model": "x", "precision": "fp16", "input_h": 388, "input_w": 518,
           "engine_path": "/nowhere/x.engine", "engine_bytes": 123,
           "engine_mtime": 99, "onnx_sha256": "abc",
           "stats": {"mean_ms": 1.5}, "device": "GPU"}
    card = viz.model_card("x", spec, run)
    for key in ("model", "input_h", "input_w", "outputs", "unit", "precision",
                "engine_fingerprint"):
        assert card[key], f"{key} missing from the card"
    assert card["unit"] == "m"
    caption = viz.caption_for(card)
    assert "388x518" in caption and "fp16" in caption
    assert card["engine_fingerprint"] in caption
    assert "point map" in caption


def test_the_fingerprint_follows_the_record_rather_than_the_model_name():
    base = {"engine_path": "/a/x.engine", "engine_bytes": 1, "engine_mtime": 2,
            "onnx_sha256": "h", "precision": "fp16", "input_h": 1, "input_w": 1}
    first, src = viz.engine_fingerprint(base)
    assert src == "record"
    assert viz.engine_fingerprint(dict(base, engine_bytes=2))[0] != first, (
        "a different engine produced the same fingerprint")
    assert viz.engine_fingerprint({})[0] == "unknown"


def test_a_fingerprint_of_a_real_file_is_a_hash_of_the_file(tmp_path):
    p = tmp_path / "x.engine"
    p.write_bytes(b"not really an engine")
    fp, src = viz.engine_fingerprint({"engine_path": str(p)})
    assert src == "engine sha256" and len(fp) == 12


def test_a_caption_line_cannot_run_into_the_next_panel():
    long = "out: " + ", ".join(f"name{i}=meaning number {i}" for i in range(12))
    lines = viz.wrap_caption(long)
    assert lines and max(len(x) for x in lines) <= viz.CAPTION_CHARS
    unbroken = viz.wrap_caption("C:/" + "a" * 300)
    assert max(len(x) for x in unbroken) <= viz.CAPTION_CHARS


# ---------------------------------------------------------------------------
# Auxiliary outputs and point clouds
# ---------------------------------------------------------------------------

def test_a_mask_is_reported_as_coverage_and_a_scalar_as_its_value():
    m = np.zeros((4, 4), np.float32)
    m[:2] = 1.0
    img, text = viz.view_of_output("mask", "valid mask", m, 4, 4, 8, 8)
    assert img.shape == (8, 8, 3)
    assert "50.0% valid" in text
    _, text = viz.view_of_output("fov_deg", "field of view in degrees",
                                 np.float32([55.0]), 4, 4, 8, 8)
    assert "55" in text


def test_point_layouts_name_models_that_still_have_the_matching_adapter():
    """The layout table is transcribed from evaluate_gt; check it still fits."""
    src = open(os.path.join(ROOT, "tools", "evaluate_gt.py"), encoding="utf-8").read()
    for model, layout in viz.POINT_LAYOUT.items():
        assert f'"{model}": {{' in src, f"{model} has no adapter any more"
        want = "_point_map_depth" if layout == "hw3" else "_point_channel_depth"
        block = src.split(f'"{model}": {{', 1)[1].split("},", 1)[0]
        assert want in block, (
            f"{model} is listed as {layout} but its adapter does not use {want}")


def test_point_map_layouts_are_not_interchangeable():
    h, w = 3, 4
    flat = np.arange(h * w * 3, dtype=np.float64)
    a = viz.points_from_point_map(flat, "hw3", h, w)
    b = viz.points_from_point_map(flat, "chw", h, w)
    assert a.shape == b.shape == (h, w, 3)
    assert not np.array_equal(a, b)
    with pytest.raises(ValueError):
        viz.points_from_point_map(flat, "whatever", h, w)


def test_unprojecting_a_constant_depth_puts_the_centre_on_the_axis():
    d = np.full((9, 9), 2.0)
    pts = viz.points_from_depth(d, viz.focal_from_fov(60.0, 9))
    assert pts[4, 4, 0] == pytest.approx(0.0)
    assert pts[4, 4, 1] == pytest.approx(0.0)
    assert np.allclose(pts[..., 2], 2.0)


def test_render_points_puts_the_near_point_in_front():
    """A painter's-algorithm scatter would draw whichever came last."""
    near = np.array([[[0.0, 0.0, 1.0]]])
    far = np.array([[[0.0, 0.0, 9.0]]])
    both = np.concatenate([near.reshape(1, -1, 3), far.reshape(1, -1, 3)], axis=1)
    colours = np.array([[[255, 0, 0], [0, 0, 255]]], np.uint8)
    img = viz.render_points(both, colours, size=(41, 41), elev_deg=0.0,
                            azim_deg=0.0, point_px=1)
    hit = img.reshape(-1, 3)
    reds = int((hit[:, 0] > 200).sum())
    blues = int((hit[:, 2] > 200).sum())
    assert reds >= 1 and blues == 0, (
        f"the far point is visible ({blues} blue pixels); the z-buffer is not "
        f"working and the back of the scene paints over the front")


def test_write_ply_is_readable_and_counts_only_valid_points(tmp_path):
    pts = np.array([[[0.0, 0.0, 1.0], [1.0, 2.0, 3.0], [0.0, 0.0, -1.0],
                     [np.nan, 0.0, 1.0]]])
    cols = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]], np.uint8)
    p = str(tmp_path / "c.ply")
    viz.write_ply(p, pts, cols)
    with open(p, "rb") as f:
        blob = f.read()
    head, body = blob.split(b"end_header\n", 1)
    assert b"element vertex 2" in head, "invalid points were written out"
    rec = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("r", "u1"), ("g", "u1"), ("b", "u1")])
    assert rec.shape == (2,)
    assert rec["z"].tolist() == [1.0, 3.0]


# ---------------------------------------------------------------------------
# Figures actually come out, headless, at three aspect ratios
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ASPECTS))
def test_a_figure_is_written_headless_for_each_aspect_ratio(tmp_path, name):
    h, w = ASPECTS[name]
    img = _image(h, w)[:, :, ::-1]
    panels = [viz.Panel(title=f"model_{i}", image=img, band="metric",
                        caption=viz.caption_for(_card(f"model_{i}")),
                        box=(4, h - 4, 4, w - 4), box_label="image")
              for i in range(5)]
    sections = [viz.Section(header="metric", panels=panels,
                            colorbar={"vmin": 0.5, "vmax": 8.0,
                                      "cmap": viz.METRIC_CMAP, "label": "m"})]
    fig = viz.render_figure(sections, f"t {name}", subtitle="s", footer="f")
    out = str(tmp_path / f"{name}.png")
    viz.save_figure(fig, out)
    assert os.path.getsize(out) > 5000
    written = cv2.imread(out)
    assert written is not None and written.shape[0] > 400


def test_an_empty_figure_does_not_crash(tmp_path):
    fig = viz.render_figure([], "nothing")
    viz.save_figure(fig, str(tmp_path / "e.png"))
    assert os.path.getsize(tmp_path / "e.png") > 0


# ---------------------------------------------------------------------------
# demo.py end to end, without a GPU
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def three_aspects(tmp_path_factory):
    d = tmp_path_factory.mktemp("aspects")
    paths = []
    for i, (name, (h, w)) in enumerate(sorted(ASPECTS.items())):
        p = d / f"aspect_{name}.png"
        cv2.imwrite(str(p), _image(h, w, i))
        paths.append(str(p))
    return paths


# Kept small on purpose: depth_pro runs at 1536x1536 and the point is the
# drawing path, not the arithmetic throughput.
DEMO_MODELS = "depth_anything_v2,zipdepth,moge_2"


def test_demo_draws_three_aspect_ratios_from_synthetic_arrays(tmp_path,
                                                              three_aspects):
    out = tmp_path / "out"
    rc = demo.main(["--synthetic", "--models", DEMO_MODELS, "--out", str(out)]
                   + [a for p in three_aspects for a in ("--image", p)])
    assert rc == 0
    for name in ASPECTS:
        for kind in ("depth", "inputs", "outputs", "cloud"):
            f = out / f"aspect_{name}_{kind}.png"
            assert f.is_file() and f.stat().st_size > 5000, f"missing {f}"
    summary = json.loads((out / "demo_summary.json").read_text())
    assert set(summary["images"]) == {f"aspect_{n}" for n in ASPECTS}
    for key, img in summary["images"].items():
        assert img["depth"]["shared_metric_range"], key
        for model, drawn in img["depth"]["per_model"].items():
            if "error" in drawn:
                continue
            assert drawn["engine_fingerprint"], model
            assert drawn["input_size"][0] and drawn["input_size"][1]
            assert drawn["unit"] in ("m", "-")


def test_saved_arrays_reproduce_the_run_they_came_from(tmp_path, three_aspects):
    """--from-saved is the path with no GPU in it, so it is the one checked."""
    images = [a for p in three_aspects for a in ("--image", p)]
    saved = tmp_path / "saved"
    first = tmp_path / "first"
    assert demo.main(["--synthetic", "--models", DEMO_MODELS,
                      "--save-outputs", str(saved), "--out", str(first),
                      "--no-pointcloud", "--no-extras",
                      "--no-inputs-figure"] + images) == 0
    assert (saved / "aspect_4x3" / "moge_2.npz").is_file()

    second = tmp_path / "second"
    assert demo.main(["--from-saved", str(saved), "--models", DEMO_MODELS,
                      "--out", str(second), "--no-pointcloud", "--no-extras",
                      "--no-inputs-figure"] + images) == 0

    a = json.loads((first / "demo_summary.json").read_text())["images"]
    b = json.loads((second / "demo_summary.json").read_text())["images"]
    assert a.keys() == b.keys()
    for key in a:
        assert (a[key]["depth"]["shared_metric_range"]
                == b[key]["depth"]["shared_metric_range"]), key
        for model in a[key]["depth"]["per_model"]:
            assert (a[key]["depth"]["per_model"][model].get("stats")
                    == b[key]["depth"]["per_model"][model].get("stats")), \
                f"{key}/{model} changed on the way through the npz"


@pytest.mark.parametrize("name", sorted(ASPECTS))
def test_every_model_puts_its_depth_back_on_the_source_grid(name, three_aspects):
    """Coordinate restoration, per model, per aspect ratio.

    A model whose output lands on the network grid instead of the source grid
    would still draw a plausible picture -- stretched, and next to correct ones.
    """
    from core import spec as spec_mod

    path = [p for p in three_aspects if p.endswith(f"aspect_{name}.png")][0]
    bgr = cv2.imread(path)
    specs = spec_mod.load_all()
    adapt, why = demo.adapters()
    assert adapt, why
    runs = demo.bench_runs()
    checked = 0
    for model in sorted(set(adapt) & set(specs)):
        card = viz.model_card(model, specs[model], runs.get(model))
        if not card["input_h"]:
            continue
        res = demo.evaluate_one(model, specs[model], card, adapt, bgr,
                                "synthetic", fx=886.81)
        assert "error" not in res, f"{model}: {res.get('error')}"
        assert res["depth"].shape == bgr.shape[:2], (
            f"{model} returned {res['depth'].shape} for a "
            f"{bgr.shape[:2]} source")
        checked += 1
    assert checked >= 8, f"only {checked} models were exercised"


def test_a_model_that_fails_is_drawn_as_a_failure_rather_than_dropped(tmp_path,
                                                                     three_aspects):
    """Rule 9. metric3d_v2 cannot produce metres without a focal length."""
    out = tmp_path / "out"
    assert demo.main(["--synthetic", "--models", "metric3d_v2,zipdepth",
                      "--out", str(out), "--no-pointcloud", "--no-extras",
                      "--no-inputs-figure", "--image", three_aspects[0]]) == 0
    per = json.loads((out / "demo_summary.json").read_text())
    per = next(iter(per["images"].values()))["depth"]["per_model"]
    assert "error" in per["metric3d_v2"]
    assert "focal" in per["metric3d_v2"]["error"]
    assert "error" not in per["zipdepth"]


def test_the_same_model_with_a_focal_length_then_succeeds(three_aspects):
    from core import spec as spec_mod

    specs = spec_mod.load_all()
    adapt, _ = demo.adapters()
    if "metric3d_v2" not in adapt:
        pytest.skip("no metric3d_v2 adapter here")
    bgr = cv2.imread(three_aspects[0])
    card = viz.model_card("metric3d_v2", specs["metric3d_v2"],
                          demo.bench_runs().get("metric3d_v2"))
    res = demo.evaluate_one("metric3d_v2", specs["metric3d_v2"], card, adapt,
                            bgr, "synthetic", fx=886.81)
    assert "error" not in res, res.get("error")
    assert res["depth"].shape == bgr.shape[:2]


def test_a_fitted_relative_model_is_drawn_with_the_values_it_was_fitted_to(
        tmp_path, three_aspects):
    """The panel has to show the fitted array, not the raw one.

    It did not: the fit was computed, recorded in the caption, and then the
    untouched disparity was coloured on the metric axis -- so a relative model
    appeared with the near object drawn as the far one, on a colourbar in
    metres, with a caption saying it had been aligned.
    """
    out = tmp_path / "out"
    assert demo.main(["--synthetic", "--models",
                      "depth_anything_v2,zipdepth,depth_anything_ac",
                      "--align-relative-to", "depth_anything_v2",
                      "--out", str(out), "--no-pointcloud", "--no-extras",
                      "--no-inputs-figure", "--image", three_aspects[0]]) == 0
    per = next(iter(json.loads(
        (out / "demo_summary.json").read_text())["images"].values()))["depth"]
    ref = per["per_model"]["depth_anything_v2"]
    assert ref["band"] == "metric"
    for model in ("zipdepth", "depth_anything_ac"):
        got = per["per_model"][model]
        assert got["band"] == "fitted", (
            "a fitted model was filed under metric; it does not claim metres")
        assert got["fit"]["reference"] == "depth_anything_v2"
        assert got["unit"] != ref["unit"], "a borrowed unit is printed as its own"
        assert got["stats"]["median"] == pytest.approx(
            ref["stats"]["median"], rel=0.15), (
            f"{model} was fitted to the reference but its drawn median "
            f"{got['stats']['median']:.3g} is nowhere near the reference's "
            f"{ref['stats']['median']:.3g}; the raw array was drawn")


def test_a_fitted_model_cannot_move_the_shared_metric_axis():
    cards = {"a": _card("a", "metric"), "f": _card("f", "fitted")}
    depths = {"a": np.linspace(1, 4, 64).reshape(8, 8),
              "f": np.linspace(50, 90, 64).reshape(8, 8)}
    ranges, note = viz.display_ranges(cards, depths)
    assert ranges["a"][:3] == ranges["f"][:3]
    assert ranges["a"][1] < 10, (
        "a model with a borrowed scale set the axis every metric model is "
        "read against")
    assert "over 1 metric models" in note


def test_no_source_of_output_is_refused_with_advice(tmp_path, three_aspects):
    with pytest.raises(SystemExit) as e:
        demo.main(["--models", "zipdepth", "--out", str(tmp_path),
                   "--image", three_aspects[0]])
    assert "--from-saved" in str(e.value) and "--live" in str(e.value)


def test_a_missing_aspect_manifest_falls_back_to_the_example(monkeypatch):
    monkeypatch.setattr(demo, "ASPECTS", os.path.join(ROOT, "no", "such.json"))
    monkeypatch.setattr(demo, "_eval_inputs_module", lambda: None)
    if not os.path.isfile(demo.EXAMPLE):
        pytest.skip("no data/example.jpg in this checkout")
    inputs, note = demo.resolve_inputs()
    assert len(inputs) == 1
    assert inputs[0]["path"] == demo.EXAMPLE
    assert "aspects.json" in note and "three" in note


def test_the_manifest_is_read_when_it_is_there(tmp_path, monkeypatch):
    """The contract with P2: a list of entries with a `file` relative to it."""
    d = tmp_path / "eval"
    (d / "aspects").mkdir(parents=True)
    entries = []
    for name, (h, w) in sorted(ASPECTS.items()):
        cv2.imwrite(str(d / "aspects" / f"{name}.png"), _image(h, w))
        entries.append({"file": f"aspects/{name}.png", "aspect": name,
                        "width": w, "height": h, "source": "test",
                        "license": "test", "sha256": "0" * 64})
    (d / "aspects.json").write_text(json.dumps(entries))
    monkeypatch.setattr(demo, "ASPECTS", str(d / "aspects.json"))
    monkeypatch.setattr(demo, "_eval_inputs_module", lambda: None)
    inputs, note = demo.resolve_inputs()
    assert len(inputs) == 3
    assert {i["aspect"] for i in inputs} == set(ASPECTS)
    assert "3 aspect ratios" in note


def test_the_synthetic_scene_is_not_flat():
    """A constant scene would make every layout test pass and prove nothing."""
    d = demo.synthetic_scene(64, 96)
    assert d.shape == (64, 96)
    assert float(d.max() - d.min()) > 2.0
    assert np.isfinite(d).all() and (d > 0).all()


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
