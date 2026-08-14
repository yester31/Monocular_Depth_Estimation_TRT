"""The P2 aspect-ratio manifest loader, including its failure modes.

core/eval_inputs.py is a contract other code reads (P7's demo among it), so
the checks that matter most here are the ones that make it fail loudly
instead of silently dropping a bad entry: a missing manifest, a missing image,
a sha256 that no longer matches the file on disk, and an entry missing one of
the required keys.
"""

import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eval_inputs import (  # noqa: E402
    DEFAULT_MANIFEST,
    EvalInputError,
    VALID_ASPECTS,
    load_eval_inputs,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_committed_manifest_loads_and_covers_three_aspects():
    entries = load_eval_inputs()
    assert len(entries) == 3
    aspects = {aspect for _, aspect, _ in entries}
    assert aspects == VALID_ASPECTS


def test_each_entry_has_the_p7_contract_keys():
    entries = load_eval_inputs()
    required = {
        "file", "aspect", "width", "height", "source", "license",
        "orig_file", "orig_width", "orig_height", "crop", "sha256",
    }
    for _, _, meta in entries:
        assert required <= meta.keys()
        assert set(meta["crop"].keys()) == {"x", "y", "w", "h"}


def test_paths_are_real_files_matching_declared_dimensions():
    from PIL import Image

    for path, _, meta in load_eval_inputs():
        assert os.path.isfile(path)
        with Image.open(path) as im:
            assert im.size == (meta["width"], meta["height"])


def test_crop_rectangle_is_reproducible_from_the_original():
    import numpy as np
    from PIL import Image

    base_dir = os.path.dirname(DEFAULT_MANIFEST)
    for _, _, meta in load_eval_inputs():
        orig_path = os.path.join(base_dir, meta["orig_file"])
        crop_path = os.path.join(base_dir, meta["file"])
        with Image.open(orig_path) as orig, Image.open(crop_path) as crop:
            assert orig.size == (meta["orig_width"], meta["orig_height"])
            c = meta["crop"]
            box = (c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"])
            rebuilt = orig.convert("RGB").crop(box)
            assert np.array_equal(np.array(rebuilt), np.array(crop.convert("RGB")))


def test_missing_manifest_fails_explicitly():
    with pytest.raises(EvalInputError, match="manifest not found"):
        load_eval_inputs(os.path.join(ROOT, "data", "eval", "does_not_exist.json"))


def test_sha256_mismatch_fails_explicitly(tmp_path):
    with open(DEFAULT_MANIFEST, "r", encoding="utf-8") as f:
        entries = json.load(f)

    base_dir = os.path.dirname(DEFAULT_MANIFEST)
    aspects_dir = tmp_path / "aspects"
    aspects_dir.mkdir()
    src = os.path.join(base_dir, entries[0]["file"])
    dst = aspects_dir / os.path.basename(entries[0]["file"])
    shutil.copyfile(src, dst)
    with open(dst, "ab") as f:
        f.write(b"\x00")  # corrupt the copy so its sha256 no longer matches
    entries[0]["file"] = f"aspects/{dst.name}"

    manifest_path = tmp_path / "aspects.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)

    with pytest.raises(EvalInputError, match="sha256 mismatch"):
        load_eval_inputs(str(manifest_path))


def test_missing_image_file_fails_explicitly(tmp_path):
    with open(DEFAULT_MANIFEST, "r", encoding="utf-8") as f:
        entries = json.load(f)
    entries[0]["file"] = "aspects/does_not_exist.png"

    manifest_path = tmp_path / "aspects.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)

    with pytest.raises(EvalInputError, match="image file not found"):
        load_eval_inputs(str(manifest_path))


def test_entry_missing_a_required_key_fails_explicitly(tmp_path):
    with open(DEFAULT_MANIFEST, "r", encoding="utf-8") as f:
        entries = json.load(f)
    del entries[0]["crop"]

    manifest_path = tmp_path / "aspects.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)

    with pytest.raises(EvalInputError, match="missing keys"):
        load_eval_inputs(str(manifest_path))


def test_manifest_must_be_a_json_array(tmp_path):
    manifest_path = tmp_path / "aspects.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"samples": []}, f)

    with pytest.raises(EvalInputError, match="expected a JSON array"):
        load_eval_inputs(str(manifest_path))
