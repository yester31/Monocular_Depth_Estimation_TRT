"""Tests for core/artifact.py — the engine plus what it needs to be usable.

The value of a manifest is that it can be trusted without opening the engine.
So the thing worth testing is not that fields get written, but that the
manifest never claims something that is not true: an engine file that is not
in the directory, a checksum that no longer matches, a build with no recorded
TensorRT version.
"""

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import artifact  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


SPEC = {
    "schema": 1, "name": "demo", "env": "demo",
    "input": {"name": "input", "rank": 4, "dtype": "float32", "layout": "NCHW"},
    "outputs": [{"name": "depth", "meaning": "relative depth"}],
    "profiles": {"bench": {"size": [518, 518]}},
    "build_targets": [{"profile": "bench", "precision": "fp16"}],
    "caveats": ["a condition worth carrying"],
}
RUN = {
    "model": "demo", "profile": "bench", "precision": "fp16",
    "device": "RTX 3080", "driver": "591.86",
    "versions": {"tensorrt": "10.16.1.11", "cuda": "12.8", "torch": "2.11.0+cu128"},
    "stats": {"mean_ms": 4.8, "fps": 208.4}, "timestamp": "2026-08-13T00:00:00",
}


def _engine(td, body=b"ENGINE-BYTES"):
    p = os.path.join(td, "model_fp16.engine")
    open(p, "wb").write(body)
    return p


def test_directory_name_carries_profile_and_precision():
    """metric3d_v2 is the only fp32 build and depth_pro the only native one, so
    a directory named for one axis collides as soon as a second is built."""
    check("profile-precision", artifact.profile_id(SPEC, RUN) == "bench-fp16",
          artifact.profile_id(SPEC, RUN))
    other = dict(RUN, precision="fp32")
    check("precision changes the id", artifact.profile_id(SPEC, other) == "bench-fp32")


def test_manifest_records_what_the_engine_cannot():
    with tempfile.TemporaryDirectory() as td:
        d = artifact.write(SPEC, RUN, _engine(td), out_root=os.path.join(td, "a"))
        man = artifact.load_manifest(d)
        check("input size present", man["input"]["size"] == [518, 518])
        check("input layout present", man["input"]["layout"] == "NCHW")
        check("output meaning present",
              man["outputs"][0]["meaning"] == "relative depth")
        check("tensorrt version present", man["built_with"]["tensorrt"] == "10.16.1.11")
        check("gpu present", man["built_with"]["gpu"] == "RTX 3080")
        check("caveats carried", man["caveats"] == ["a condition worth carrying"])
        check("measurement carried", man["measured"]["fps"] == 208.4)


def test_engine_is_copied_and_verifies():
    with tempfile.TemporaryDirectory() as td:
        d = artifact.write(SPEC, RUN, _engine(td), out_root=os.path.join(td, "a"))
        check("engine copied", os.path.isfile(os.path.join(d, "model.engine")))
        check("verifies clean", artifact.verify(d) == [], str(artifact.verify(d)))


def test_no_copy_does_not_claim_a_file_that_is_absent():
    """The bug this test exists for: with copying declined the manifest kept
    saying file: model.engine while the directory held only the manifest."""
    with tempfile.TemporaryDirectory() as td:
        d = artifact.write(SPEC, RUN, _engine(td), out_root=os.path.join(td, "a"),
                           copy_engine=False)
        man = artifact.load_manifest(d)
        check("no engine in the directory",
              not os.path.isfile(os.path.join(d, "model.engine")))
        check("manifest does not claim one", man["engine"]["file"] is None,
              str(man["engine"]["file"]))
        check("says why", man["engine"].get("reason") == "not copied",
              str(man["engine"].get("reason")))
        check("still records the checksum", bool(man["engine"]["sha256"]))


def test_missing_source_is_recorded_not_invented():
    with tempfile.TemporaryDirectory() as td:
        d = artifact.write(SPEC, RUN, os.path.join(td, "nope.engine"),
                           out_root=os.path.join(td, "a"))
        man = artifact.load_manifest(d)
        check("file is null", man["engine"]["file"] is None)
        check("reason recorded",
              man["engine"].get("reason") == "source not on this machine")
        check("no invented checksum", man["engine"]["sha256"] is None)


def test_verify_catches_a_tampered_engine():
    with tempfile.TemporaryDirectory() as td:
        d = artifact.write(SPEC, RUN, _engine(td), out_root=os.path.join(td, "a"))
        open(os.path.join(d, "model.engine"), "wb").write(b"SOMETHING-ELSE!!")
        problems = artifact.verify(d)
        check("mismatch detected", any("sha256" in p or "size" in p for p in problems),
              str(problems))


def test_verify_catches_a_missing_engine():
    with tempfile.TemporaryDirectory() as td:
        d = artifact.write(SPEC, RUN, _engine(td), out_root=os.path.join(td, "a"))
        os.remove(os.path.join(d, "model.engine"))
        check("absence detected", artifact.verify(d) != [])


def test_verify_flags_an_unrecorded_tensorrt():
    """An engine whose builder version is unknown may fail to deserialize
    elsewhere with nothing to explain why."""
    with tempfile.TemporaryDirectory() as td:
        run = dict(RUN, versions={})
        d = artifact.write(SPEC, run, _engine(td), out_root=os.path.join(td, "a"))
        check("missing tensorrt flagged",
              any("TensorRT" in p for p in artifact.verify(d)),
              str(artifact.verify(d)))


def test_artifact_module_needs_no_gpu_stack():
    """Assembling or checking an artifact must work anywhere, or it will only
    ever be done on the one machine that can also build engines."""
    import ast
    src = open(os.path.join(ROOT, "core", "artifact.py"), encoding="utf-8").read()
    imported = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    heavy = imported & {"torch", "tensorrt", "numpy", "cv2", "onnx"}
    check("no torch/tensorrt/numpy", not heavy, str(sorted(heavy)))


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
