"""Does core/pointmap.py agree with MoGe's own solver on real point maps?

    python tools/check_pointmap.py --manifest data/eval/diode_indoors.json \
        --root C:/Users/soy/data/diode/val --limit 10

core/pointmap.py exists to remove an import from a model's clone. The fair
objection to reimplementing anything is that it becomes a second thing to be
wrong, and unit tests against synthetic cameras do not answer it -- they check
the algorithm, not the agreement.

This runs the engine on real images and puts both implementations on the same
point map. It needs the clone, so it is a tool rather than a unit test; the
clone being required *here* is exactly the dependency being removed from the
evaluation.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    # model -> (height, width, mask output index)
    "moge_2": (388, 518, 2),
    "metric_anything": (388, 518, 1),
}


def upstream_solver():
    import torch  # noqa: F401  (the upstream function is a torch one)

    for path, module in (
            (os.path.join(ROOT, "models", "moge_2"), "MoGe.moge.utils.geometry_torch"),
            (os.path.join(ROOT, "models", "metric_anything", "metric_anything",
                          "models", "student_pointmap"), "moge.utils.geometry_torch")):
        if not os.path.isdir(path):
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            mod = __import__(module, fromlist=["recover_focal_shift"])
            return mod.recover_focal_shift
        except Exception:                                     # noqa: BLE001
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="largest shift difference that still counts as agreement")
    args = ap.parse_args()

    import torch

    import common
    import evaluate_gt as ev
    from core import bench, pointmap

    theirs = upstream_solver()
    if theirs is None:
        print("MoGe's solver is not importable here; nothing to compare against")
        return 1

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    runs = {r["model"]: r for r in bench.load_all(
        os.path.join(ROOT, "reports", "bench"))}

    worst_overall = 0.0
    for model, (h, w, mask_idx) in MODELS.items():
        if model not in runs or not os.path.exists(runs[model].get("engine_path", "")):
            print(f"{model}: no engine here, skipped")
            continue
        engine = ev.load_engine(runs[model]["engine_path"])
        diffs, better = [], []
        with engine, engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            for s in manifest["samples"][:args.limit]:
                bgr = cv2.imread(os.path.join(args.root, s["rgb"]))
                if bgr is None:
                    continue
                inputs[0].host = ev.ADAPTERS[model]["pre"](bgr)
                raw = common.do_inference(context, engine=engine, bindings=bindings,
                                          inputs=inputs, outputs=outputs, stream=stream)
                points = np.asarray(raw[0]).reshape(1, h, w, 3).copy()
                mask = np.asarray(raw[mask_idx]).reshape(1, h, w).copy() > 0.5

                _, their_shift = theirs(torch.from_numpy(points),
                                        torch.from_numpy(mask))
                their_shift = float(their_shift.reshape(-1)[0])
                _, my_shift = pointmap.recover_focal_shift(points[0], mask[0])
                diffs.append(abs(their_shift - my_shift))
                # Where they differ, which one is actually at the bottom? A
                # disagreement is only a defect if the reimplementation is the
                # worse solution; upstream stops at ftol=1e-3, which is loose
                # enough to leave a basin early.
                if diffs[-1] > 1e-4:
                    uv = pointmap.normalized_view_plane_uv(w, h, np.float32)
                    pl = pointmap._downsample_nearest(points[0])
                    ul = pointmap._downsample_nearest(uv)
                    ml = pointmap._downsample_nearest(mask[0])
                    pl, ul = pl[ml], ul[ml]
                    uvf = np.asarray(ul, np.float64).reshape(-1, 2)
                    xy = np.asarray(pl, np.float64)[..., :2].reshape(-1, 2)
                    z = np.asarray(pl, np.float64)[..., 2].reshape(-1)

                    def cost(sh):
                        proj = xy / (z + sh)[:, None]
                        f = (proj * uvf).sum() / np.square(proj).sum()
                        return float(np.square(f * proj - uvf).sum())

                    better.append((cost(my_shift), cost(their_shift)))
            common.free_buffers(inputs, outputs, stream)

        if not diffs:
            print(f"{model}: no images read")
            continue
        worst = max(diffs)
        worst_overall = max(worst_overall, worst)
        print(f"{model:18} {len(diffs)} images   max |shift difference| "
              f"{worst:.3e}   median {np.median(diffs):.3e}   "
              f"{'ok' if worst <= args.tol else 'DISAGREES'}")
        if better:
            mine_wins = sum(1 for a, b in better if a < b)
            print(f"{'':18} of {len(better)} frames past 1e-4: this "
                  f"implementation has the lower residual on {mine_wins}, "
                  f"upstream on {len(better) - mine_wins}")
            for a, b in better:
                print(f"{'':20} residual  mine {a:.6e}   upstream {b:.6e}")

    print(f"\n{'agrees' if worst_overall <= args.tol else 'DISAGREEMENT'} "
          f"within {args.tol:g}")
    return 0 if worst_overall <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
