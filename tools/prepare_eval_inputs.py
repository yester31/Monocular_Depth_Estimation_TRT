"""Regenerate the multi-aspect-ratio evaluation crops and their manifest.

    python tools/prepare_eval_inputs.py            # write data/eval/aspects.json
    python tools/prepare_eval_inputs.py --check     # regenerate and diff only

The three crops (4:3, 16:9, portrait) are not hand-picked photos; they are
fixed pixel rectangles cut from DIODE indoor frames that already live in this
repo under data/eval/aspects/originals/. Nothing here downloads anything or
depends on an external dataset path at run time.

manifest is not hand-written. This script is the only thing that writes
data/eval/aspects.json -- every entry's sha256 is computed from the crop file
this run just produced, so a manifest that disagrees with the crops on disk
cannot exist. Re-running with --check must be a no-op: same originals, same
crop rectangles, same bytes out, hence the same sha256, every time.
"""

import argparse
import hashlib
import json
import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASPECTS_DIR = os.path.join(ROOT, "data", "eval", "aspects")
MANIFEST_PATH = os.path.join(ROOT, "data", "eval", "aspects.json")

LICENSE = (
    "MIT -- 'The DIODE dataset and the code is released using the MIT "
    "license.' Stated at https://diode-dataset.org/, checked 2026-08-13 "
    "(same source cited in tools/make_eval_manifest.py)."
)
DIODE_URL = "http://diode-dataset.s3.amazonaws.com/val.tar.gz"

# Each spec names a DIODE original already committed under
# data/eval/aspects/originals/ and a fixed, centred crop rectangle. Originals
# are natively 1024x768 (4:3) DIODE indoor RGB frames, so every crop below is
# a sub-rectangle -- nothing is ever upsampled.
#
# orig_diode_id is the sample id in data/eval/diode_indoors.json this frame
# was copied from (read-only reference for provenance; this script does not
# load that file).
SPECS = [
    {
        "aspect": "4:3",
        "orig_file": "originals/00019_00183_indoors_000_010.png",
        "orig_diode_id": "indoors/scene_00019/scan_00183/00019_00183_indoors_000_010",
        "crop": {"x": 0, "y": 0, "w": 1024, "h": 768},
        "out_file": "aspect_4x3.png",
    },
    {
        "aspect": "16:9",
        "orig_file": "originals/00020_00186_indoors_220_040.png",
        "orig_diode_id": "indoors/scene_00020/scan_00186/00020_00186_indoors_220_040",
        # 1024 wide, height trimmed to 16:9 (576 = 1024 * 9 / 16), centred.
        "crop": {"x": 0, "y": 96, "w": 1024, "h": 576},
        "out_file": "aspect_16x9.png",
    },
    {
        "aspect": "portrait",
        "orig_file": "originals/00021_00192_indoors_320_020.png",
        "orig_diode_id": "indoors/scene_00021/scan_00192/00021_00192_indoors_320_020",
        # 768 tall, width trimmed to 3:4 (576 = 768 * 3 / 4), centred.
        "crop": {"x": 224, "y": 0, "w": 576, "h": 768},
        "out_file": "aspect_portrait.png",
    },
]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_entry(spec, write):
    orig_path = os.path.join(ASPECTS_DIR, spec["orig_file"])
    if not os.path.isfile(orig_path):
        raise FileNotFoundError(f"missing DIODE original: {orig_path}")

    im = Image.open(orig_path).convert("RGB")
    ow, oh = im.size
    c = spec["crop"]
    if c["x"] < 0 or c["y"] < 0 or c["x"] + c["w"] > ow or c["y"] + c["h"] > oh:
        raise ValueError(
            f"crop out of bounds for {spec['orig_file']}: {c} vs {ow}x{oh} original"
        )
    box = (c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"])
    crop = im.crop(box)

    out_path = os.path.join(ASPECTS_DIR, spec["out_file"])
    if write:
        # Fixed compression, no metadata -- keeps the bytes (and therefore
        # the sha256) identical across reruns and machines.
        crop.save(out_path, format="PNG", compress_level=6)
    elif not os.path.isfile(out_path):
        raise FileNotFoundError(
            f"missing crop output: {out_path} (run without --check once first)"
        )

    return {
        "file": f"aspects/{spec['out_file']}",
        "aspect": spec["aspect"],
        "width": crop.size[0],
        "height": crop.size[1],
        "source": (
            f"DIODE val/indoors, sample {spec['orig_diode_id']} "
            f"({DIODE_URL}, https://diode-dataset.org/)"
        ),
        "license": LICENSE,
        "orig_file": f"aspects/{spec['orig_file']}",
        "orig_width": ow,
        "orig_height": oh,
        "crop": c,
        "sha256": sha256(out_path),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="regenerate crops in memory only (they must already exist on "
        "disk) and diff the resulting manifest against data/eval/aspects.json "
        "instead of writing it",
    )
    args = ap.parse_args()

    entries = [build_entry(spec, write=not args.check) for spec in SPECS]

    if args.check:
        if not os.path.isfile(MANIFEST_PATH):
            print(f"no manifest at {MANIFEST_PATH} to check against")
            return 1
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            committed = json.load(f)
        if committed != entries:
            print("aspects.json does not match a fresh rebuild from the originals:")
            print(json.dumps(entries, indent=2, ensure_ascii=False))
            return 1
        print(f"aspects.json matches a fresh rebuild ({len(entries)} entries)")
        return 0

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"{len(entries)} entries -> {os.path.relpath(MANIFEST_PATH, ROOT)}")
    for e in entries:
        print(f"  {e['aspect']:>8}  {e['width']}x{e['height']}  {e['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
