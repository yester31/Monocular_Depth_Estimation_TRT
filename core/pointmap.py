"""Turning a point map into depth, without needing the model's repository.

moge_2 and metric_anything do not return depth. They return a point map whose
Z is depth only after a shift has been recovered -- the network is free to
place the scene anywhere along the optical axis, and the shift is what pins it
down. MoGe solves for it, and the evaluation used to import that solver from
whichever clone happened to be on the machine.

That worked and was fragile in a way worth removing: the clone is a moving
target this repository does not control, it is excluded from version control
on purpose, and an evaluation that cannot run without it cannot run on a
machine that only has engines.

Reimplemented here instead, and the objection to reimplementing -- that it
adds a second thing to be wrong -- is answered by tests/test_pointmap.py,
which runs both against the same point maps and requires them to agree. A
reimplementation checked against the original is not a second source of truth;
it is the same one with the dependency removed.

Transcribed from MoGe's geometry_torch.recover_focal_shift and
geometry_numpy.solve_optimal_focal_shift, commit 3fab839f, the one this
repository already pins for utils3d.
"""

import numpy as np

DOWNSAMPLE = (64, 64)


def normalized_view_plane_uv(width, height, dtype=np.float32):
    """The view-plane grid MoGe solves against.

    Corners at (+-w/diagonal, +-h/diagonal), so the grid is scaled by the
    diagonal rather than by either side -- which is what makes the recovered
    focal independent of aspect ratio. Sample centres, not edges: the span is
    multiplied by (n-1)/n.
    """
    aspect = width / height
    span_x = aspect / (1 + aspect ** 2) ** 0.5
    span_y = 1.0 / (1 + aspect ** 2) ** 0.5
    u = np.linspace(-span_x * (width - 1) / width,
                    span_x * (width - 1) / width, width, dtype=dtype)
    v = np.linspace(-span_y * (height - 1) / height,
                    span_y * (height - 1) / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing="xy")
    return np.stack([u, v], axis=-1)


def _nearest_indices(n_in, n_out):
    """The indices torch's F.interpolate(mode='nearest') keeps.

    Not np.round: torch's nearest takes floor(i * n_in / n_out), which for an
    even downsample picks the *upper-left* of each block rather than the
    nearest centre. Rounding instead moves the sample by one pixel and changes
    the recovered shift in the fourth decimal -- small, and exactly the kind of
    difference that makes a reimplementation disagree for no visible reason.
    """
    return (np.arange(n_out) * (n_in / n_out)).astype(np.int64)


def _downsample_nearest(a, size=DOWNSAMPLE):
    rows = _nearest_indices(a.shape[0], size[0])
    cols = _nearest_indices(a.shape[1], size[1])
    return a[rows][:, cols]


def solve_optimal_focal_shift(uv, xyz):
    """min over shift of |f(shift) * xy / (z + shift) - uv|, f in closed form.

    One free parameter. For any shift the best focal is the projection of uv
    onto the reprojected points, so only the shift is searched -- which is why
    a one-dimensional solver is enough for something that looks like a
    two-parameter fit.
    """
    from scipy.optimize import least_squares

    uv = np.asarray(uv).reshape(-1, 2)
    xy = np.asarray(xyz)[..., :2].reshape(-1, 2)
    z = np.asarray(xyz)[..., 2].reshape(-1)

    def residual(shift):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return (f * xy_proj - uv).ravel()

    # x0=0, ftol=1e-3, Levenberg-Marquardt: upstream's settings, not defaults.
    # A tighter tolerance finds a slightly different shift, so matching it is
    # part of matching the answer.
    sol = least_squares(residual, x0=0, ftol=1e-3, method="lm")
    shift = np.float32(np.squeeze(sol["x"]))
    xy_proj = xy / (z + shift)[:, None]
    focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
    return shift, focal


def recover_focal_shift(points, mask=None, downsample_size=DOWNSAMPLE):
    """(focal, shift) for one point map of shape (H, W, 3).

    `focal` is relative to the half-diagonal of the map, as upstream defines
    it. Solved on a 64x64 nearest-neighbour downsample, which is upstream's
    choice and not an approximation this adds: the full map is a quarter of a
    million points for a two-parameter fit.
    """
    pts = np.asarray(points)
    h, w = pts.shape[-3], pts.shape[-2]
    uv = normalized_view_plane_uv(w, h, dtype=pts.dtype)

    pts_lr = _downsample_nearest(pts, downsample_size)
    uv_lr = _downsample_nearest(uv, downsample_size)
    if mask is not None:
        m_lr = _downsample_nearest(np.asarray(mask, dtype=bool), downsample_size)
        pts_lr, uv_lr = pts_lr[m_lr], uv_lr[m_lr]

    if uv_lr.reshape(-1, 2).shape[0] < 2:
        # Upstream's answer when there is nothing to fit. Not an error: a frame
        # the model masked out entirely still has to produce a number.
        return 1.0, 0.0
    shift, focal = solve_optimal_focal_shift(uv_lr, pts_lr)
    return float(focal), float(shift)


def depth_from_point_map(points, mask=None, metric_scale=None):
    """Point-map Z shifted into camera space, and scaled if the model says so."""
    _, shift = recover_focal_shift(points, mask)
    depth = np.asarray(points)[..., 2] + shift
    if metric_scale is not None:
        depth = depth * float(np.asarray(metric_scale).reshape(-1)[0])
    return depth
