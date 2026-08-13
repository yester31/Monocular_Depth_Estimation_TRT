"""Package an engine together with everything needed to use it.

    artifacts/<model>/<profile>-<precision>/
      model.engine
      manifest.json

An engine on its own is unusable and dangerous. It does not record what input
layout it wants, what its outputs mean, what resolution it was built for, or
which TensorRT built it -- and TensorRT will happily deserialize a plan built
by a different version only to fail later, or not at all. Every one of those
facts already exists in this repository, scattered across spec.json, the
benchmark record and the build log; the manifest is where they travel together.

The plan for this repository is explicit that ONNX `metadata_props` is not the
place for it and that shipping a bare engine is not allowed. This module is
that rule made executable.

Nothing here imports torch or TensorRT: an artifact is assembled from files
and JSON, so it can be built, inspected or verified anywhere.
"""

import hashlib
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")

SCHEMA = 1


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def profile_id(spec, run):
    """The directory name for one build: profile and precision.

    Both matter and neither is implied by the other -- metric3d_v2 is the only
    fp32 build, and depth_pro the only native-profile one, so a directory named
    for just one of them would collide the day a second precision is built.
    """
    t = spec["build_targets"][0]
    prof = (run or {}).get("profile") or t.get("profile", "bench")
    prec = (run or {}).get("precision") or t.get("precision", "fp32")
    return f"{prof}-{prec}"


def build_manifest(spec, run, engine_path, onnx_path=None):
    """Everything a caller needs that the engine itself does not carry."""
    h, w = spec["profiles"][
        (run or {}).get("profile") or spec["build_targets"][0]["profile"]]["size"]
    versions = (run or {}).get("versions") or {}
    man = {
        "schema": SCHEMA,
        "model": spec["name"],
        "profile": (run or {}).get("profile") or spec["build_targets"][0]["profile"],
        "precision": (run or {}).get("precision")
                     or spec["build_targets"][0]["precision"],
        "input": dict(spec["input"], size=[h, w]),
        "outputs": spec["outputs"],
        "caveats": list(spec.get("caveats", [])),
        "engine": {
            "file": "model.engine",
            "sha256": sha256(engine_path) if os.path.isfile(engine_path) else None,
            "bytes": os.path.getsize(engine_path) if os.path.isfile(engine_path) else None,
            "built_from": os.path.basename(onnx_path) if onnx_path else None,
            "onnx_sha256": sha256(onnx_path) if onnx_path and os.path.isfile(onnx_path) else None,
        },
        "built_with": {
            "tensorrt": versions.get("tensorrt"),
            "cuda": versions.get("cuda"),
            "torch": versions.get("torch"),
            "gpu": (run or {}).get("device"),
            "driver": (run or {}).get("driver"),
        },
        "measured": ((run or {}).get("stats") or None),
        "measured_at": (run or {}).get("timestamp"),
    }
    return man


def write(spec, run, engine_path, onnx_path=None, out_root=None, copy_engine=True):
    """Assemble one artifact directory. Returns its path."""
    out_root = out_root or ARTIFACTS
    d = os.path.join(out_root, spec["name"], profile_id(spec, run))
    os.makedirs(d, exist_ok=True)

    man = build_manifest(spec, run, engine_path, onnx_path)
    present = os.path.isfile(engine_path)

    # `file` names what is in this directory, and must be null whenever
    # nothing is. Two ways that happens: the engine is on another machine, or
    # copying was declined. The checksum and size are still recorded when the
    # source is readable -- they describe the engine, not the copy -- but
    # claiming a file that is not here is exactly the lie verify() exists to
    # catch, and an earlier version of this function wrote it.
    if copy_engine and present:
        shutil.copy2(engine_path, os.path.join(d, "model.engine"))
    else:
        man["engine"]["file"] = None
        man["engine"]["source"] = engine_path
        man["engine"]["reason"] = ("not copied" if present
                                   else "source not on this machine")

    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return d


def load_manifest(path):
    p = path if path.endswith(".json") else os.path.join(path, "manifest.json")
    with open(p, encoding="utf-8") as f:
        man = json.load(f)
    if man.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema {man.get('schema')}, expected {SCHEMA}")
    return man


def verify(dir_path):
    """Check an artifact against its own manifest. Returns a list of problems."""
    problems = []
    try:
        man = load_manifest(dir_path)
    except Exception as e:                              # noqa: BLE001
        return [f"manifest unreadable: {e}"]

    engine = man.get("engine", {})
    name = engine.get("file")
    if not name:
        problems.append("manifest records no engine file")
        return problems
    p = os.path.join(dir_path, name)
    if not os.path.isfile(p):
        problems.append(f"{name} missing")
        return problems
    if engine.get("bytes") is not None and os.path.getsize(p) != engine["bytes"]:
        problems.append("engine size differs from the manifest")
    if engine.get("sha256") and sha256(p) != engine["sha256"]:
        problems.append("engine sha256 differs from the manifest")
    if not man.get("built_with", {}).get("tensorrt"):
        problems.append("no TensorRT version recorded — the engine may not load "
                        "on another install and nothing here would say why")
    return problems
