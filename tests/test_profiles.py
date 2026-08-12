"""Keep onnx_export.py and onnx2trt.py agreeing on the input size.

The engine input shape is static and the resolution is part of the model name,
so a mismatch means onnx2trt.py looks for an ONNX that was never written. That
is exactly what happened to moge_2 (exported 291x518, loaded 388x518).

Also checks the models that carry a bench/native profile switch: both files
must offer the same profiles and the same size for each.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


def code_of(rel):
    """File contents with comments and docstrings stripped."""
    path = os.path.join(ROOT, rel.replace("/", os.sep))
    if not os.path.exists(path):
        return None
    src = open(path, encoding="utf-8").read()
    src = re.sub(r'"""[\s\S]*?"""', '""', src)
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))


def sizes_in(code):
    """Every (h, w) the file pins, from either spelling."""
    out = []
    for m in re.finditer(r'input_h\s*,\s*input_w\s*=\s*\(?\s*(\d+)\s*,\s*(\d+)', code):
        out.append((int(m.group(1)), int(m.group(2))))
    h = re.search(r'^\s*input_h\s*=\s*(\d+)', code, re.M)
    w = re.search(r'^\s*input_w\s*=\s*(\d+)', code, re.M)
    if h and w:
        out.append((int(h.group(1)), int(w.group(1))))
    return out


# models whose two scripts must pin the same size
PAIRS = [
    "Depth_Anything_V2", "Depth_Anything_AC", "Depth_Anything_V3",
    "Distill_Any_Depth", "Metric3D_V2", "Metric_Anything", "MoGe_2",
    "StreamVGGT", "Uni_Depth_V2", "UniK3D", "VGGT",
]

# Depth_Pro is excluded on purpose: it spells its size as `img_size = 1536`
# for both dimensions rather than input_h/input_w, so sizes_in() finds
# nothing to compare. Checked separately below.

# Models offering both a bench and a native profile.
#
# Only depth_anything_ac qualifies. unidepth_v2 and unik3d were tried and
# reverted: they pick their own input size from the source aspect ratio, so no
# single native size exists — see the WARNING at the top of their onnx2trt.py
# and docs/model_contracts.md 5.7. They ship bench only.
PROFILED = ["Depth_Anything_AC"]

# Models pinned to 518x518 with no native variant, where the metric output is
# known to differ from upstream. Each must say so in its own source.
BENCH_ONLY_WITH_CAVEAT = ["Uni_Depth_V2", "UniK3D"]


def test_export_and_trt_agree():
    for model in PAIRS:
        exp, trt = code_of(f"{model}/onnx_export.py"), code_of(f"{model}/onnx2trt.py")
        if exp is None or trt is None:
            check(f"{model} both scripts exist", False, "missing file")
            continue
        se, st = set(sizes_in(exp)), set(sizes_in(trt))
        if not se or not st:
            check(f"{model} sizes found", False, f"export={se} trt={st}")
            continue
        check(f"{model} sizes agree", se == st, f"export={sorted(se)} trt={sorted(st)}")


def test_profiled_models_offer_both():
    for model in PROFILED:
        for script in ("onnx_export.py", "onnx2trt.py"):
            code = code_of(f"{model}/{script}")
            has_switch = bool(re.search(r"profile\s*=\s*'(bench|native)'", code))
            has_both = "'bench'" in code and "'native'" in code
            check(f"{model}/{script} has a profile switch", has_switch and has_both)


def test_bench_is_518_square():
    """The comparison profile has to be identical across models."""
    for model in PROFILED:
        code = code_of(f"{model}/onnx2trt.py")
        m = re.search(r"\(518,\s*518\)\s*if\s*profile\s*==\s*'bench'", code)
        check(f"{model} bench is 518x518", bool(m))


def test_bench_only_models_document_the_caveat():
    """A model whose metric output does not match upstream must say so where a
    reader will see it, not only in the docs."""
    for model in BENCH_ONLY_WITH_CAVEAT:
        for script in ("onnx_export.py", "onnx2trt.py"):
            src = open(os.path.join(ROOT, model, script), encoding="utf-8").read()
            if script == "onnx2trt.py":
                check(f"{model}/{script} warns about metric depth",
                      "WARNING" in src and "metric" in src.lower())
            else:
                check(f"{model}/{script} points at the explanation",
                      "onnx2trt.py" in src)
        # and the profile switch really is gone.
        #
        # Must be a standalone assignment: `profile='bench'` also appears as a
        # keyword argument to bench.record(), which is just labelling the
        # result file and is not a switch that changes the input size.
        for script in ("onnx_export.py", "onnx2trt.py"):
            src = open(os.path.join(ROOT, model, script), encoding="utf-8").read()
            check(f"{model}/{script} has no profile switch",
                  not re.search(r"^\s*profile\s*=\s*'[a-z]+'\s*(#.*)?$", src, re.M))


def test_depth_pro_size():
    """Depth_Pro uses one `img_size` for both dimensions and takes it from the
    model on the export side, so it needs its own check rather than an
    exception in PAIRS.

    What matters is that the size reaches the ONNX filename on both sides.
    Without it, an export at one size and a load at another does not fail as a
    missing file -- it builds an engine whose buffers silently disagree.
    """
    exp, trt = code_of("Depth_Pro/onnx_export.py"), code_of("Depth_Pro/onnx2trt.py")
    for name, code in (("export", exp), ("trt", trt)):
        check(f"Depth_Pro {name} puts the size in the model name",
              bool(re.search(r'model_name\s*=\s*f?"depth_pro_\{img_size\}x\{img_size\}"',
                             code)), code[:0])
    check("Depth_Pro trt pins img_size",
          bool(re.search(r"^\s*img_size\s*=\s*\d+", trt, re.M)))
    check("Depth_Pro export takes img_size from the model",
          "img_size = model.img_size" in exp)


def test_metric_anything_keeps_the_source_aspect():
    """D13: a square input makes the derived intrinsics square, giving a 4:3
    photo a 1:1 field of view. It must stay rectangular, and match MoGe_2,
    which runs the identical MoGe post-process."""
    trt = code_of("Metric_Anything/onnx2trt.py")
    sizes = set(sizes_in(trt))
    check("metric_anything is not square",
          all(h != w for h, w in sizes), str(sorted(sizes)))
    check("metric_anything matches moge_2",
          sizes == set(sizes_in(code_of("MoGe_2/onnx2trt.py"))),
          f"metric_anything={sorted(sizes)} "
          f"moge_2={sorted(set(sizes_in(code_of('MoGe_2/onnx2trt.py'))))}")


def _resolved_name(path, drop_sim):
    """The model_name string a file actually produces, or None if undecidable.

    Comparing the source lines does not work: the two scripts often reach the
    same name by different routes (one f-string versus three appends), and
    that is fine. What matters is the string. So this replays the simple
    literal assignments and the model_name chain in an empty namespace.

    Returns None when a value comes from something other than a literal --
    Depth_Pro takes its size from `model.img_size`, which cannot be evaluated
    without loading the model. Those are checked separately.
    """
    import ast as _ast
    src = open(path, encoding="utf-8").read()
    tree = _ast.parse(src)

    fn = next((n for n in tree.body
               if isinstance(n, _ast.FunctionDef) and n.name == "main"), None)
    if fn is None:
        return None

    class _Dtypes:
        """Stand-in for torch, so `dtype = torch.half` and the
        `... if dtype == torch.half else ...` that follows both resolve.
        Identity comparison is all these expressions need."""
        def __getattr__(self, name):
            return f"<torch.{name}>"

    env, saw_name = {"torch": _Dtypes()}, False

    def value_of(node):
        """Literal if possible, otherwise evaluate against what we have."""
        try:
            return _ast.literal_eval(node)
        except Exception:
            # See the note on eval below: repo source, no builtins, only
            # previously collected literals in scope.
            return eval(_ast.unparse(node), {"__builtins__": {}}, env)

    for node in fn.body:
        if not isinstance(node, _ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, _ast.Tuple):          # input_h, input_w = 388, 518
            try:
                values = value_of(node.value)
            except Exception:
                continue
            names = [e.id for e in target.elts if isinstance(e, _ast.Name)]
            if len(names) == len(values):
                env.update(dict(zip(names, values)))
            continue
        if not isinstance(target, _ast.Name):
            continue

        if target.id == "model_name":
            line = _ast.unparse(node.value)
            if drop_sim and "_sim" in line:
                continue                             # trt picks the simplified graph
            # eval, not ast.literal_eval: the whole point is to resolve an
            # f-string over previously seen literals, which literal_eval
            # rejects. The input is this repository's own source, the only
            # names in scope are the literals collected above, and builtins
            # are removed -- so nothing here can call out.
            try:
                env["model_name"] = eval(line, {"__builtins__": {}}, env)
            except Exception:
                return None
            saw_name = True
        else:
            try:
                env[target.id] = value_of(node.value)
            except Exception:
                pass                                 # not resolvable; may not be needed

    return env.get("model_name") if saw_name else None


def test_model_name_resolves_the_same():
    """The ONNX filename is the contract between the two scripts.

    onnx2trt.py loads whatever onnx_export.py wrote, by name. A mismatch is
    either a confusing missing-file error or, if the names happen to collide,
    an engine quietly built from the wrong graph. moge_2 hit the first
    (exported 291x518, loaded 388x518); depth_pro was one edit from the second
    because its name carried no size at all.

    This also catches the latent case where the two files agree today only
    because a flag is False on both sides -- flip `dynamo` in onnx_export.py
    alone and VGGT writes vggt_..._dynamo.onnx while onnx2trt.py still looks
    for vggt_....onnx.
    """
    for model in PAIRS:
        e = _resolved_name(os.path.join(ROOT, model, "onnx_export.py"), False)
        t = _resolved_name(os.path.join(ROOT, model, "onnx2trt.py"), True)
        if e is None or t is None:
            print(f"  ..    {model} model_name not statically decidable")
            continue
        check(f"{model} model_name matches", e == t,
              f"export={e!r} trt={t!r}")


def test_moge_regression():
    """The mismatch that motivated this file."""
    exp, trt = code_of("MoGe_2/onnx_export.py"), code_of("MoGe_2/onnx2trt.py")
    check("moge_2 export matches trt", set(sizes_in(exp)) == set(sizes_in(trt)),
          f"export={sorted(set(sizes_in(exp)))} trt={sorted(set(sizes_in(trt)))}")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
