"""Pick an evaluation subset from a DIODE download and write it down exactly.

    python tools/make_eval_manifest.py --root C:/Users/soy/data/diode/val \
        --split indoors --count 50 --out data/eval/diode_indoors.json

The dataset itself is 2.6 GB and stays outside the repository; what goes in is
this manifest -- which files were used, in what order, with a SHA-256 each. A
metric computed over "DIODE indoors" is not reproducible, because the next
person picks a different subset and gets a different number. A metric computed
over this file is.

Selection is a fixed stride through the sorted list rather than a random
sample, so it spreads across scenes, has no seed to record, and produces the
same subset on any machine.

Units are not taken on faith. DIODE's devkit does not state them, so the tool
reads the selected depth maps and reports the distribution; the indoor split
lands around 6-10 m median with a 17 m maximum and the outdoor around 28 m
with 78 m, which is metres and nothing else.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE = {
    "dataset": "DIODE",
    "split_file": "val.tar.gz",
    "url": "http://diode-dataset.s3.amazonaws.com/val.tar.gz",
    "homepage": "https://diode-dataset.org/",
    "license": "MIT",
    "license_note": (
        "Stated on the project site: 'The DIODE dataset and the code is "
        "released using the MIT license.' Checked 2026-08-13."
    ),
    "citation": "DIODE: A Dense Indoor and Outdoor DEpth Dataset, arXiv:1908.00463",
    # One global calibration for the whole dataset, from the devkit's
    # intrinsics.txt. It is here because a canonical-depth model needs a real
    # focal length to produce metres at all -- metric3d_v2 was written up as
    # permanently blocked for want of one, which is true of an arbitrary photo
    # and false of a dataset that ships its camera.
    "intrinsics": {"fx": 886.81, "fy": 927.06, "cx": 512.0, "cy": 384.0,
                   "source": "diode-devkit intrinsics.txt; reproduces the "
                             "paper's stated 60x45 degree design field of view"},
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def entries(root, split):
    """Every (rgb, depth, mask) triple in one split, in a stable order."""
    found = []
    pattern = os.path.join(root, split, "**", "*_depth.npy")
    for depth in sorted(glob.glob(pattern, recursive=True)):
        stem = depth[: -len("_depth.npy")]
        rgb, mask = stem + ".png", stem + "_depth_mask.npy"
        if os.path.isfile(rgb) and os.path.isfile(mask):
            found.append((rgb, depth, mask))
    return found


def stride_sample(items, count):
    if count <= 0 or count >= len(items):
        return items
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the extracted val/ directory")
    ap.add_argument("--split", default="indoors", choices=("indoors", "outdoor"))
    ap.add_argument("--count", type=int, default=50, help="0 for every sample")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-depth", type=float, default=None,
                    help="ground truth beyond this is not scored")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    all_items = entries(root, args.split)
    if not all_items:
        print(f"no samples under {root}/{args.split}")
        return 1
    chosen = stride_sample(all_items, args.count)

    samples, valid_fracs, medians, maxima = [], [], [], []
    for rgb, depth, mask in chosen:
        d = np.load(depth)
        m = np.load(mask).astype(bool)
        d = d[..., 0] if d.ndim == 3 else d
        m = m[..., 0] if m.ndim == 3 else m
        v = d[m & (d > 0)]
        valid_fracs.append(float(m.mean()))
        medians.append(float(np.median(v)) if v.size else float("nan"))
        maxima.append(float(v.max()) if v.size else float("nan"))
        samples.append({
            "id": os.path.relpath(rgb, root).replace("\\", "/")[:-4],
            "rgb": os.path.relpath(rgb, root).replace("\\", "/"),
            "depth": os.path.relpath(depth, root).replace("\\", "/"),
            "mask": os.path.relpath(mask, root).replace("\\", "/"),
            "height": int(d.shape[0]),
            "width": int(d.shape[1]),
            "valid_fraction": round(float(m.mean()), 4),
            "sha256": {"rgb": sha256(rgb), "depth": sha256(depth),
                       "mask": sha256(mask)},
        })

    manifest = {
        "source": SOURCE,
        "split": args.split,
        "selection": (f"every {len(all_items) / len(chosen):.2f}th of "
                      f"{len(all_items)} sorted samples"),
        "count": len(samples),
        "depth_units": "metres",
        "depth_convention": "z-depth",
        "depth_convention_evidence": (
            "measured, not documented: no DIODE source states whether the "
            "stored scalar is z-depth or range along the ray, and the two "
            "differ by up to 22.7% at a corner. Back-projecting both ways and "
            "fitting planes to flat surfaces gives a flatness residual that "
            "stays put under the z reading and grows with patch area under the "
            "range reading -- 10.3x apart at 160px. See "
            "tools/check_diode_convention.py"),
        "depth_units_evidence": (
            f"median of per-sample medians {np.nanmedian(medians):.2f}, "
            f"largest valid depth {np.nanmax(maxima):.2f}, over the selected "
            f"samples -- read from the arrays, not from documentation"),
        "invalid": "mask == 0, and depth <= 0",
        "max_depth": args.max_depth,
        "mean_valid_fraction": round(float(np.mean(valid_fracs)), 4),
        "samples": samples,
    }
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"{len(samples)} samples -> {os.path.relpath(out, ROOT)}")
    print(f"valid pixels {np.mean(valid_fracs) * 100:.1f}% on average; "
          f"depth median {np.nanmedian(medians):.2f} m, max {np.nanmax(maxima):.2f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
