"""Ground-truth metrics, checked against answers worked out by hand.

The reason these get their own tests: every number this project publishes from
here on is one of these three, and a metric that is quietly wrong is worse than
no metric -- it looks like evidence. So each case has an answer that can be
derived on paper rather than by running the code and blessing what comes out.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gt import align, metrics, policy_for  # noqa: E402


def test_a_perfect_prediction_scores_perfectly():
    g = np.array([[1.0, 2.0], [4.0, 8.0]])
    m = np.ones_like(g, dtype=bool)
    r = metrics(g, g, m)
    assert r["n"] == 4
    assert r["abs_rel"] == pytest.approx(0.0)
    assert r["rmse"] == pytest.approx(0.0)
    assert r["delta1"] == pytest.approx(1.0)


def test_a_uniform_ten_percent_error():
    g = np.array([[10.0, 20.0]])
    p = g * 1.1
    m = np.ones_like(g, dtype=bool)
    r = metrics(p, g, m)
    assert r["abs_rel"] == pytest.approx(0.1)
    # RMSE is in metres, so it is dominated by the far pixel: sqrt((1^2+2^2)/2)
    assert r["rmse"] == pytest.approx(np.sqrt(2.5))
    # 1.1 < 1.25, so every pixel is within delta1
    assert r["delta1"] == pytest.approx(1.0)


def test_delta1_counts_the_ratio_both_ways():
    g = np.array([4.0, 4.0, 4.0, 4.0])
    p = np.array([4.0, 5.5, 2.0, 4.9])          # ratios 1.0, 1.375, 2.0, 1.225
    m = np.ones_like(g, dtype=bool)
    r = metrics(p, g, m)
    assert r["delta1"] == pytest.approx(0.5)     # 1.0 and 1.225
    assert r["delta2"] == pytest.approx(0.75)    # + 1.375, since 1.25^2 = 1.5625


def test_invalid_pixels_are_not_scored():
    g = np.array([1.0, 2.0, 0.0, 4.0])
    p = np.array([1.0, 2.0, 99.0, 4.0])
    m = np.array([True, True, True, False])
    r = metrics(p, g, m)
    # The zero-depth pixel is dropped even though the mask allows it, and the
    # masked-out pixel is dropped even though its depth is fine.
    assert r["n"] == 2
    assert r["abs_rel"] == pytest.approx(0.0)


def test_max_depth_drops_far_ground_truth():
    g = np.array([1.0, 50.0, 300.0])
    p = g.copy()
    m = np.ones_like(g, dtype=bool)
    assert metrics(p, g, m)["n"] == 3
    assert metrics(p, g, m, max_depth=80.0)["n"] == 2


def test_nothing_valid_returns_a_count_and_no_scores():
    g = np.array([1.0, 2.0])
    r = metrics(g, g, np.zeros(2, dtype=bool))
    assert r == {"n": 0}


def test_scale_alignment_recovers_a_pure_scale():
    g = np.array([2.0, 4.0, 6.0])
    p = g / 3.0
    out, fit = align(p, g, np.ones(3, dtype=bool), "scale")
    assert fit["scale"] == pytest.approx(3.0)
    assert out == pytest.approx(g)


def test_scale_shift_recovers_an_affine_disparity():
    # This fixture IS a disparity-like model, so it says so. The parameter is
    # not a convenience: which way round a relative model's output runs decides
    # whether the fit is affine in the space the model was trained in.
    g = np.array([1.0, 2.0, 4.0, 8.0])
    disparity = 1.0 / g
    p = (disparity - 0.05) / 3.0                 # invert: disp = 3*p + 0.05
    out, fit = align(p, g, np.ones(4, dtype=bool), "scale_shift",
                     pred_is_inverse=True)
    assert fit["scale"] == pytest.approx(3.0)
    assert fit["shift"] == pytest.approx(0.05)
    assert out == pytest.approx(g)


def test_alignment_is_fitted_only_where_the_mask_allows():
    g = np.array([2.0, 4.0, 6.0, 1000.0])
    p = np.array([1.0, 2.0, 3.0, 3.0])           # last pixel is nonsense
    m = np.array([True, True, True, False])
    _, fit = align(p, g, m, "scale")
    assert fit["scale"] == pytest.approx(2.0)


def test_no_alignment_leaves_the_prediction_alone():
    p = np.array([1.0, 2.0])
    out, fit = align(p, np.array([5.0, 5.0]), np.ones(2, dtype=bool), "none")
    assert out == pytest.approx(p)
    assert fit == {}


def test_a_constant_prediction_cannot_be_scale_shifted():
    # No slope is determined by the data. Refused rather than fitted to
    # whatever the numerics happen to produce.
    with pytest.raises(ValueError):
        align(np.full(4, 0.5), np.array([1.0, 2.0, 3.0, 4.0]),
              np.ones(4, dtype=bool), "scale_shift")


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        align(np.ones(2), np.ones(2), np.ones(2, dtype=bool), "whatever")


def test_the_contract_decides_the_alignment():
    assert policy_for("metric") == "none"
    assert policy_for("relative") == "scale_shift"
    # The one that matters: an output nobody has pinned down must not acquire a
    # policy just because scale_shift would produce a number for it.
    assert policy_for("unknown") is None
    assert policy_for("canonical") is None


def test_scale_shift_on_a_model_that_returns_depth():
    # The fit is affine in disparity. A relative model that returns depth has
    # to be inverted before it, not fitted through it: 1/(s*d + t) is not a
    # hyperbola in d, and forcing the line onto one lands on a negative slope
    # that reduces the error slightly and means nothing.
    g = np.array([1.0, 2.0, 4.0, 8.0])
    pred_depth = g * 7.0                      # right shape, wrong unit
    out, fit = align(pred_depth, g, np.ones(4, dtype=bool), "scale_shift",
                     pred_is_inverse=False)
    assert fit["scale"] == pytest.approx(7.0)
    assert fit["shift"] == pytest.approx(0.0, abs=1e-9)
    assert out == pytest.approx(g)


def test_the_two_orientations_disagree_and_the_wrong_one_is_worse():
    g = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    pred_depth = g * 3.0
    right, _ = align(pred_depth, g, np.ones(5, dtype=bool), "scale_shift",
                     pred_is_inverse=False)
    wrong, _ = align(pred_depth, g, np.ones(5, dtype=bool), "scale_shift",
                     pred_is_inverse=True)
    assert metrics(right, g, np.ones(5, dtype=bool))["abs_rel"] < 1e-9
    assert metrics(wrong, g, np.ones(5, dtype=bool))["abs_rel"] > 0.1


def test_orientation_reports_which_way_a_prediction_runs():
    from core.gt import orientation
    g = np.array([1.0, 2.0, 4.0, 8.0])
    m = np.ones(4, dtype=bool)
    assert orientation(g * 2.0, g, m) > 0.99          # returns depth
    assert orientation(1.0 / g, g, m) < -0.5          # returns disparity
