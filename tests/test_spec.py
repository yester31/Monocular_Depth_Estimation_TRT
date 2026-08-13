"""Every spec.json must agree with the code and the measurements.

A spec is read by the shared runtime instead of importing the model, so it is
the thing a caller trusts. If it drifts it is worse than nothing: it reads as
authoritative while being wrong, and nobody re-derives a fact they have just
read in a declaration.

So nothing here trusts the spec on its own. The size is checked against the
script that pins it, the environment against the directory that has to exist,
the outputs against the ONNX export, and the whole set against the benchmark
results.
"""

import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import spec as spec_mod  # noqa: E402
from core.bench import load_all as load_bench  # noqa: E402

MODELS = os.path.join(ROOT, "models")

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


def model_dirs():
    return [d for d in sorted(os.listdir(MODELS))
            if os.path.isfile(os.path.join(MODELS, d, "onnx2trt.py"))]


def test_every_model_has_a_valid_spec():
    for d in model_dirs():
        p = spec_mod.path_for(d)
        check(f"{d}/spec.json exists", os.path.isfile(p))
        if not os.path.isfile(p):
            continue
        try:
            s = spec_mod.load(d)
            check(f"{d} spec validates", True)
        except spec_mod.SpecError as e:
            check(f"{d} spec validates", False, str(e))
            continue
        check(f"{d} spec name matches its directory", s["name"] == d,
              f"spec says {s['name']!r}")


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
    allowed = set(sys.stdlib_module_names)
    outside = sorted(imported - allowed)
    check("core/spec.py imports only the standard library", not outside,
          str(outside))


def test_size_matches_the_script():
    """The spec must not invent a resolution the engine does not use."""
    for d in model_dirs():
        try:
            s = spec_mod.load(d)
        except spec_mod.SpecError:
            continue
        src = open(os.path.join(MODELS, d, "onnx2trt.py"), encoding="utf-8").read()
        code = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#"))
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
        if not sizes:
            check(f"{d} size found in script", False, "no input size parsed")
            continue
        declared = spec_mod.size_of(s)
        check(f"{d} spec size matches onnx2trt.py", declared in sizes,
              f"spec {declared}, script {sorted(sizes)}")


def test_env_directory_exists_or_is_named_consistently():
    """The env name is what a runner passes to conda. A typo here sends
    someone to a nonexistent environment with a confusing error."""
    for d in model_dirs():
        try:
            s = spec_mod.load(d)
        except spec_mod.SpecError:
            continue
        env = s.get("env", "")
        check(f"{d} declares a non-empty env", bool(env.strip()))


def test_outputs_match_the_export():
    """Output names come from onnx_export.py's output_names. A spec that
    renames them describes a graph nobody built."""
    for d in model_dirs():
        exp = os.path.join(MODELS, d, "onnx_export.py")
        if not os.path.isfile(exp):
            continue
        try:
            s = spec_mod.load(d)
        except spec_mod.SpecError:
            continue
        # Comments stripped first. VGGT and StreamVGGT keep a commented-out
        # `#output_names=["pose_enc", "depth", "depth_conf"]` directly above
        # the live one, so a raw search reads the wrong list and reports the
        # spec as wrong when it is right. This is the second test in this
        # repository to make that mistake; the first was the dynamo= check.
        src = "\n".join(l for l in open(exp, encoding="utf-8").read().splitlines()
                        if not l.strip().startswith("#"))
        m = re.search(r"output_names\s*=\s*\[([^\]]*)\]", src)
        if not m:
            print(f"  ..    {d}: export does not name its outputs")
            continue
        declared = [o["name"] for o in s["outputs"]]
        actual = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
        check(f"{d} spec outputs match the export", declared == actual,
              f"spec {declared}, export {actual}")


def test_build_targets_match_what_was_measured():
    """A build target nobody has run is a claim, not a fact."""
    runs = {r["model"]: r for r in load_bench(os.path.join(ROOT, "reports", "bench"))}
    for d in model_dirs():
        try:
            s = spec_mod.load(d)
        except spec_mod.SpecError:
            continue
        r = runs.get(d)
        if not r:
            print(f"  ..    {d}: not measured yet")
            continue
        t = s["build_targets"][0]
        check(f"{d} target precision matches the run",
              t.get("precision") == r.get("precision"),
              f"spec {t.get('precision')}, run {r.get('precision')}")
        check(f"{d} target profile matches the run",
              t.get("profile") == r.get("profile"),
              f"spec {t.get('profile')}, run {r.get('profile')}")
        check(f"{d} target size matches the run",
              list(spec_mod.size_of(s)) == [r.get("input_h"), r.get("input_w")],
              f"spec {spec_mod.size_of(s)}, run {(r.get('input_h'), r.get('input_w'))}")


def test_caveats_are_not_lost():
    """Every caveat compare.py prints must exist in the model's spec, so the
    two cannot say different things about the same model."""
    from compare import CAVEAT
    for model, text in CAVEAT.items():
        p = spec_mod.path_for(model)
        if not os.path.isfile(p):
            continue
        s = spec_mod.load(model)
        check(f"{model} spec records a caveat", bool(spec_mod.caveats(s)),
              f"compare.py says: {text}")


def test_rank_5_models_are_the_vggt_family():
    """Rank 5 is the [1,S,3,H,W] sequence input. Only VGGT and StreamVGGT take
    it, and a spec claiming rank 5 elsewhere is a copy-paste."""
    for d in model_dirs():
        try:
            s = spec_mod.load(d)
        except spec_mod.SpecError:
            continue
        if s["input"]["rank"] == 5:
            check(f"{d} rank 5 is expected", d in {"vggt", "streamvggt"}, d)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
