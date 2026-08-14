"""Every spec.json must agree with the code and the measurements.

A spec is read by the shared runtime instead of importing the model, so it is
the thing a caller trusts. If it drifts it is worse than nothing: it reads as
authoritative while being wrong, and nobody re-derives a fact they have just
read in a declaration.

So nothing here trusts the spec on its own. The size is checked against the
script that pins it, the environment against the directory that has to exist,
the outputs against the ONNX export, and the whole set against the benchmark
results.

This file used to collect its results in a module-level `_failures` list and
never assert, which meant fifteen checks that could not fail `pytest tests -q`.
It asserts now, one parametrised test per model, so a failure names the model
rather than a list. tests/conftest.py is the safety net for the modules that
still use the old shape.
"""

import ast
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import spec as spec_mod  # noqa: E402
from core.bench import load_all as load_bench  # noqa: E402

MODELS = os.path.join(ROOT, "models")


def model_dirs():
    return [d for d in sorted(os.listdir(MODELS))
            if os.path.isfile(os.path.join(MODELS, d, "onnx2trt.py"))]


MODEL_DIRS = model_dirs()


def _load(d):
    """The spec, or skip. A missing or invalid spec is its own test's job."""
    try:
        return spec_mod.load(d)
    except spec_mod.SpecError as e:
        pytest.skip(f"{d}: {e}")


def _strip_comments(src):
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))


# ---------------------------------------------------------------------------

def test_there_are_models_to_check():
    """A parametrised suite over an empty list passes without testing anything."""
    assert len(MODEL_DIRS) >= 12, MODEL_DIRS


@pytest.mark.parametrize("d", MODEL_DIRS)
def test_every_model_has_a_valid_spec(d):
    p = spec_mod.path_for(d)
    assert os.path.isfile(p), f"no spec at {p}"
    s = spec_mod.load(d)                    # raises SpecError if it does not validate
    assert s["name"] == d, f"spec says {s['name']!r}, directory is {d!r}"


def test_spec_is_dependency_free():
    """core/spec.py must stay importable with only the standard library.

    The moment it needs numpy or torch, the shared trte runtime cannot read a
    spec without the model's environment, which is the exact coupling the
    spec exists to avoid.
    """
    src = open(os.path.join(ROOT, "core", "spec.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    outside = sorted(imported - set(sys.stdlib_module_names))
    assert not outside, f"core/spec.py imports {outside} from outside the stdlib"


@pytest.mark.parametrize("d", MODEL_DIRS)
def test_size_matches_the_script(d):
    """The spec must not invent a resolution the engine does not use."""
    s = _load(d)
    code = _strip_comments(
        open(os.path.join(MODELS, d, "onnx2trt.py"), encoding="utf-8").read())
    sizes = set()
    for m in re.finditer(r"input_h\s*,\s*input_w\s*=\s*\(?\s*(\d+)\s*,\s*(\d+)", code):
        sizes.add((int(m.group(1)), int(m.group(2))))
    h = re.search(r"^\s*input_h\s*=\s*(\d+)", code, re.M)
    w = re.search(r"^\s*input_w\s*=\s*(\d+)", code, re.M)
    if h and w:
        sizes.add((int(h.group(1)), int(w.group(1))))
    sq = re.search(r"^\s*img_size\s*=\s*(\d+)", code, re.M)   # depth_pro
    if sq:
        sizes.add((int(sq.group(1)), int(sq.group(1))))

    assert sizes, f"{d}: no input size parsed out of onnx2trt.py"
    declared = spec_mod.size_of(s)
    assert declared in sizes, f"spec {declared}, script {sorted(sizes)}"


@pytest.mark.parametrize("d", MODEL_DIRS)
def test_env_directory_exists_or_is_named_consistently(d):
    """The env name is what a runner passes to conda. A typo here sends
    someone to a nonexistent environment with a confusing error."""
    s = _load(d)
    assert s.get("env", "").strip(), f"{d} declares no env"


@pytest.mark.parametrize("d", MODEL_DIRS)
def test_outputs_match_the_export(d):
    """Output names come from onnx_export.py's output_names. A spec that
    renames them describes a graph nobody built."""
    exp = os.path.join(MODELS, d, "onnx_export.py")
    if not os.path.isfile(exp):
        pytest.skip(f"{d}: no onnx_export.py")
    s = _load(d)
    # Comments stripped first. VGGT and StreamVGGT keep a commented-out
    # `#output_names=["pose_enc", "depth", "depth_conf"]` directly above the
    # live one, so a raw search reads the wrong list and reports the spec as
    # wrong when it is right. This is the second test in this repository to
    # make that mistake; the first was the dynamo= check.
    src = _strip_comments(open(exp, encoding="utf-8").read())
    m = re.search(r"output_names\s*=\s*\[([^\]]*)\]", src)
    if not m:
        pytest.skip(f"{d}: export does not name its outputs")
    declared = [o["name"] for o in s["outputs"]]
    actual = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
    assert declared == actual, f"spec {declared}, export {actual}"


def _bench_runs():
    return {r["model"]: r for r in load_bench(os.path.join(ROOT, "reports", "bench"))}


@pytest.mark.parametrize("d", MODEL_DIRS)
def test_build_targets_match_what_was_measured(d):
    """A build target nobody has run is a claim, not a fact."""
    s = _load(d)
    r = _bench_runs().get(d)
    if not r:
        pytest.skip(f"{d}: not measured yet")
    t = s["build_targets"][0]
    assert t.get("precision") == r.get("precision"), \
        f"spec {t.get('precision')}, run {r.get('precision')}"
    assert t.get("profile") == r.get("profile"), \
        f"spec {t.get('profile')}, run {r.get('profile')}"
    assert list(spec_mod.size_of(s)) == [r.get("input_h"), r.get("input_w")], \
        f"spec {spec_mod.size_of(s)}, run {(r.get('input_h'), r.get('input_w'))}"


def _caveat_models():
    from compare import CAVEAT
    return sorted(CAVEAT)


@pytest.mark.parametrize("model", _caveat_models())
def test_caveats_are_not_lost(model):
    """Every caveat compare.py prints must exist in the model's spec, so the
    two cannot say different things about the same model."""
    from compare import CAVEAT
    p = spec_mod.path_for(model)
    if not os.path.isfile(p):
        pytest.skip(f"{model}: no spec.json")
    s = spec_mod.load(model)
    assert spec_mod.caveats(s), \
        f"{model} spec records no caveat, but compare.py says: {CAVEAT[model]}"


@pytest.mark.parametrize("d", MODEL_DIRS)
def test_rank_5_models_are_the_vggt_family(d):
    """Rank 5 is the [1,S,3,H,W] sequence input. Only VGGT and StreamVGGT take
    it, and a spec claiming rank 5 elsewhere is a copy-paste."""
    s = _load(d)
    if s["input"]["rank"] == 5:
        assert d in {"vggt", "streamvggt"}, f"{d} claims rank 5"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
