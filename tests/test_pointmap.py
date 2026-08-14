"""The point-map solver, against cases whose answer is known.

core/pointmap.py replaces an import from a model's clone. The objection to
doing that is that a reimplementation is a second thing to be wrong, so these
tests do two things: check the algorithm against point maps built from a known
camera, where the right shift is the one that was put in; and, where the clone
is present, check it against MoGe's own function on the same data
(tools/check_pointmap.py, which needs the clone and so is not a unit test).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pointmap import (  # noqa: E402
    depth_from_point_map, normalized_view_plane_uv, recover_focal_shift)


def synthetic(h, w, focal, shift, seed=0):
    """A point map from a real camera, then pushed along Z by `shift`.

    Built by inverting exactly what the solver fits: uv = focal * xy / z. So
    the shift the solver should recover is -shift, the amount that undoes the
    push.
    """
    rng = np.random.default_rng(seed)
    uv = normalized_view_plane_uv(w, h, dtype=np.float64)
    z = 2.0 + rng.random((h, w))              # a scene 2-3 units away
    xy = uv * (z / focal)[..., None]
    pts = np.concatenate([xy, z[..., None]], axis=-1)
    pts[..., 2] -= shift                       # the network's arbitrary offset
    return pts.astype(np.float32)


def test_the_uv_grid_is_scaled_by_the_diagonal():
    uv = normalized_view_plane_uv(4, 4)
    # Square: the two spans match, and the corner sits at 1/sqrt(2) scaled by
    # (n-1)/n because these are sample centres.
    assert uv[0, 0, 0] == pytest.approx(-(1 / 2 ** 0.5) * 3 / 4, rel=1e-6)
    assert uv[0, 0, 1] == pytest.approx(-(1 / 2 ** 0.5) * 3 / 4, rel=1e-6)
    wide = normalized_view_plane_uv(8, 4)
    assert wide[..., 0].max() > wide[..., 1].max()   # wider than tall


def test_a_point_map_with_no_shift_recovers_none():
    pts = synthetic(64, 64, focal=1.0, shift=0.0)
    focal, shift = recover_focal_shift(pts)
    assert shift == pytest.approx(0.0, abs=1e-3)
    assert focal == pytest.approx(1.0, rel=1e-2)


@pytest.mark.parametrize("applied", [0.5, -0.4, 1.25])
def test_the_recovered_shift_undoes_the_applied_one(applied):
    pts = synthetic(64, 64, focal=1.2, shift=applied)
    _, shift = recover_focal_shift(pts)
    assert shift == pytest.approx(applied, abs=2e-2)


def test_the_focal_comes_back_too():
    for f in (0.8, 1.0, 1.6):
        pts = synthetic(64, 64, focal=f, shift=0.3)
        focal, _ = recover_focal_shift(pts)
        assert focal == pytest.approx(f, rel=2e-2)


def test_depth_is_z_plus_the_shift_times_the_scale():
    pts = synthetic(64, 64, focal=1.0, shift=0.6)
    depth = depth_from_point_map(pts, metric_scale=np.array([3.0]))
    # The scene was 2-3 units away and is scaled by 3.
    assert depth.min() > 5.5 and depth.max() < 9.5


def test_a_mask_that_leaves_nothing_still_answers():
    # Upstream returns (1, 0) rather than raising, and a frame the model masked
    # out entirely still has to produce a number rather than stop the run.
    pts = synthetic(64, 64, focal=1.0, shift=0.2)
    focal, shift = recover_focal_shift(pts, mask=np.zeros((64, 64), dtype=bool))
    assert (focal, shift) == (1.0, 0.0)


def test_the_mask_is_honoured():
    pts = synthetic(64, 64, focal=1.0, shift=0.3)
    corrupt = pts.copy()
    corrupt[32:, :, :] = 99.0                  # half the map is nonsense
    mask = np.ones((64, 64), dtype=bool)
    mask[32:, :] = False
    _, shift = recover_focal_shift(corrupt, mask=mask)
    assert shift == pytest.approx(0.3, abs=3e-2)


def test_nearest_downsampling_matches_torchs_rule():
    from core.pointmap import _nearest_indices
    # floor(i * n_in / n_out), not round: torch's nearest picks the upper-left
    # of each block. For 4 -> 2 that is 0 and 2, where rounding gives 0 and 2
    # as well, but for 6 -> 4 they differ.
    assert list(_nearest_indices(4, 2)) == [0, 2]
    assert list(_nearest_indices(6, 4)) == [0, 1, 3, 4]
