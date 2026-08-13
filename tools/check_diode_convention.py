"""Does DIODE store z-depth or range along the ray?

    python tools/check_diode_convention.py [patch_px]

**Answer, measured 2026-08-14: z-depth. Use the arrays as they are.** The
numbers are at the bottom of this docstring; the test is kept so the claim can
be re-run rather than believed.

Why it had to be asked: no DIODE source states the convention. The paper says
only "we then set the depth of pixel p_ij to the robust mean of the depth of
points in X_ij" and never defines "the depth of points". The devkit never
back-projects, so it offers no formula either, and an open devkit issue (#20)
reports that the published intrinsics give visibly distorted point clouds --
which is what feeding range into a z-depth back-projection looks like.

It matters because the two differ by sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1):
1.000 at the centre, 1.227 at a corner. Every model here predicts z-depth, so
the wrong reading puts up to 23% of systematic error into the ground-truth
comparison -- an order of magnitude above the differences between the models
being compared, and shaped like a radial gradient, so it would flatter models
that blur the periphery.

The test needs no extra data: back-project each depth map both ways and ask
which reading makes flat things flat. A wall is flat in the world; under the
wrong convention it bows, and the bow grows towards the frame edges where the
conventions diverge.

Two details decide whether it works.

Patch size. A plane fit absorbs the entire first-order difference between the
conventions, so a small patch is nearly blind -- only the second-order
curvature survives. At 24 px the two readings are 1.05x apart and the test says
nothing; the separation is what grows with patch size, and that growth is
itself the evidence:

    patch    as z-depth     as range     ratio    patches
     24 px     0.0066%       0.0069%      1.05x    19294
     48 px     0.0087%       0.0149%      1.7x      4263
     96 px     0.0115%       0.0518%      4.5x       667
    160 px     0.0140%       0.1444%     10.3x       120

(median flatness residual as a fraction of depth, 28 indoor images.) The
z-depth residual barely moves as the patch grows -- that is real surface
texture. The range residual grows with area, which is an unmodelled curvature.

Patch selection. Patches are kept if they are flat under EITHER reading;
selecting by one convention's own residual would decide the answer in advance.
Patches near the principal point are skipped because the two readings agree
there and such a patch votes for nothing.
"""
import glob, os
import numpy as np

ROOT = r"C:/Users/soy/data/diode/val"
# From the devkit's intrinsics.txt. These reproduce the paper's stated 60x45
# degree design field of view, so the ray geometry is almost certainly right.
FX, FY = 886.81, 927.06
import sys
PATCH = int(sys.argv[1]) if len(sys.argv)>1 else 24
MIN_R = 260          # px from the principal point; below this the test is blind


def backproject(depth, is_range):
    h, w = depth.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x, y = (u - cx) / FX, (v - cy) / FY
    z = depth / np.sqrt(x * x + y * y + 1.0) if is_range else depth
    return np.stack([x * z, y * z, z], -1), np.hypot(u - cx, v - cy)


def plane_rms(pts):
    """RMS distance to the best-fit plane, relative to the patch's mean depth."""
    c = pts.mean(0)
    q = pts - c
    n = np.linalg.svd(q, full_matrices=False)[2][-1]
    return float(np.sqrt(np.mean((q @ n) ** 2)) / max(c[2], 1e-6))


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "indoors", "**", "*_depth.npy"),
                             recursive=True))[::12]
    tot = {"z": [], "range": []}
    for f in files:
        d = np.load(f)
        d = d[..., 0] if d.ndim == 3 else d
        m = np.load(f.replace("_depth.npy", "_depth_mask.npy")).astype(bool)
        m = m[..., 0] if m.ndim == 3 else m
        m &= d > 0
        clouds = {k: backproject(d, k == "range") for k in ("z", "range")}
        h, w = d.shape
        for y in range(0, h - PATCH, PATCH):
            for x in range(0, w - PATCH, PATCH):
                sel = m[y:y + PATCH, x:x + PATCH]
                if sel.mean() < 0.98:
                    continue
                r = clouds["z"][1][y:y + PATCH, x:x + PATCH].mean()
                if r < MIN_R:
                    continue
                got = {}
                for k, (pts, _) in clouds.items():
                    p = pts[y:y + PATCH, x:x + PATCH][sel]
                    if np.ptp(p[:, 2]) / max(p[:, 2].mean(), 1e-6) > 0.25:
                        got = {}
                        break
                    got[k] = plane_rms(p)
                # Keep the patch only if SOMETHING flat is there, judged by the
                # better of the two, so the selection cannot favour either.
                if got and min(got.values()) < 0.02:
                    for k in got:
                        tot[k].append(got[k])
    n = len(tot["z"])
    print(f"{len(files)} images, {n} planar patches beyond {MIN_R}px from centre")
    if not n:
        return
    z, rg = np.array(tot["z"]), np.array(tot["range"])
    print(f"  read as z-depth : median flatness residual {np.median(z)*100:.4f}% of depth")
    print(f"  read as range   : median flatness residual {np.median(rg)*100:.4f}% of depth")
    better = (rg < z).mean()
    print(f"  patches flatter under the range reading: {better*100:.1f}%")
    print(f"  verdict: DIODE stores "
          f"{'RANGE (convert before use)' if better > 0.6 else 'Z-DEPTH (use as is)' if better < 0.4 else 'UNRESOLVED by this test'}")


main()
