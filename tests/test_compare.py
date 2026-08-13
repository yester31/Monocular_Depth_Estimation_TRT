"""Tests for compare.py — the generated comparison table.

The thing worth guarding is not the formatting but the honesty: models run at
different input sizes must not end up in one ranked list, and an output that
is not metric must not be labelled metric.
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.bench import Bench, save  # noqa: E402
from compare import FOLDER, KIND, render  # noqa: E402
from core.bench import load_all  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


def seed(td, specs):
    for model, h, w, prec in specs:
        save(Bench(model=model, samples_ms=[10.0, 11.0, 12.0], precision=prec,
                   input_h=h, input_w=w, device="RTX 3080", warmup=10,
                   versions={"tensorrt": "10.13.3.9"},
                   timestamp="2026-08-13T10:00:00"), td)
    return load_all(td)


def test_empty_says_so():
    txt = render([])
    check("explains emptiness", "No benchmark results yet" in txt)
    check("no fake table", "| model" not in txt)


def test_sizes_are_separated():
    """A 518 model and a 1536 model in one ranked table would read as a
    quality comparison. They are not."""
    with tempfile.TemporaryDirectory() as td:
        txt = render(seed(td, [("depth_anything_v2", 518, 518, "fp16"),
                               ("depth_pro", 1536, 1536, "fp16")]))
        check("518 section", "### Input 518x518" in txt)
        check("1536 section", "### Input 1536x1536" in txt)
        check("warns across sections", "not** comparable" in txt, txt[-400:])


def test_single_size_omits_the_warning():
    with tempfile.TemporaryDirectory() as td:
        txt = render(seed(td, [("depth_anything_v2", 518, 518, "fp16"),
                               ("unik3d", 518, 518, "fp16")]))
        check("no cross-section warning", "not** comparable" not in txt)
        check("both models listed",
              "`depth_anything_v2`" in txt and "`unik3d`" in txt)


def test_caveats_follow_the_model():
    with tempfile.TemporaryDirectory() as td:
        txt = render(seed(td, [("metric3d_v2", 616, 1064, "fp16")]))
        check("D12 surfaced", "D12" in txt, txt)
        check("not called metric", "canonical" in txt)
        # a caveat for an absent model must not appear
        check("no unik3d caveat", "3.15x" not in txt)


def test_missing_models_are_named():
    with tempfile.TemporaryDirectory() as td:
        txt = render(seed(td, [("depth_anything_v2", 518, 518, "fp16")]))
        check("lists what is absent", "## Not measured" in txt)
        check("names a missing model", "`vggt`" in txt)
        check("does not list the present one",
              txt.split("## Not measured")[1].count("depth_anything_v2") == 0)


def test_mixed_hardware_is_flagged():
    """Two GPUs in one table is the single easiest way to publish a wrong
    ranking, so it has to be impossible to miss."""
    with tempfile.TemporaryDirectory() as td:
        runs = seed(td, [("depth_anything_v2", 518, 518, "fp16")])
        other = dict(runs[0])
        other["device"] = "NVIDIA RTX 4090"
        other["model"] = "unik3d"
        txt = render(runs + [other])
        check("mixed hardware warned", "Mixed hardware" in txt, txt[:600])


def test_precisions_both_listed():
    with tempfile.TemporaryDirectory() as td:
        txt = render(seed(td, [("depth_anything_v2", 518, 518, "fp16"),
                               ("depth_anything_v2", 518, 518, "fp32")]))
        check("fp16 row", "| fp16 " in txt)
        check("fp32 row", "| fp32 " in txt)


def test_kind_table_covers_every_model_folder():
    """A model folder with no KIND entry would render as '?' in the output
    column, which is how a table quietly stops being trustworthy."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models = os.path.join(root, "models")
    folders = {d for d in os.listdir(models)
               if os.path.isfile(os.path.join(models, d, "onnx2trt.py"))}
    # A key maps to its folder either by an explicit FOLDER entry or by the
    # lowercase convention every other model already follows.
    resolved = {k: FOLDER.get(k, k) for k in KIND}
    lower = {f.lower(): f for f in folders}

    unknown = sorted(f for f in folders
                     if f not in resolved.values() and f.lower() not in
                     {v.lower() for v in resolved.values()})
    check("every model folder has a KIND entry", not unknown, str(unknown))

    dangling = sorted(k for k, f in resolved.items()
                      if f not in folders and f.lower() not in lower)
    check("no KIND entry for a folder that is gone", not dangling, str(dangling))


def test_folder_overrides_are_still_needed():
    """FOLDER exists only for folders whose name does not match the model key.
    An entry that has become redundant should be deleted, not left to rot."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for key, folder in FOLDER.items():
        check(f"{key} override points at a real folder",
              os.path.isdir(os.path.join(root, "models", folder)), folder)
        check(f"{key} override is not redundant", folder.lower() != key,
              f"{folder} already lowercases to {key} - drop the entry")


def test_readmes_are_up_to_date():
    """The READMEs carry generated numbers, so they can go stale silently.

    This is the whole point of the markers: every figure a reader sees came
    from a result file. A README that quotes a measurement by hand is wrong
    from the next run onward, which is what the twelve of them used to do.
    """
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(ROOT, "compare.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    check("compare.py --check passes", r.returncode == 0,
          (r.stdout + r.stderr).strip()[-400:])


def test_every_model_readme_has_markers():
    """A model folder whose README has no BENCH block would quietly never show
    a measurement, and compare.py would not complain loudly enough."""
    models = os.path.join(ROOT, "models")
    for d in sorted(os.listdir(models)):
        p = os.path.join(models, d, "onnx2trt.py")
        if not os.path.isfile(p):
            continue
        readme = os.path.join(models, d, "README.md")
        check(f"{d}/README.md exists", os.path.isfile(readme))
        if not os.path.isfile(readme):
            continue
        src = open(readme, encoding="utf-8").read()
        check(f"{d}/README.md has BENCH markers",
              "<!-- BENCH:BEGIN -->" in src and "<!-- BENCH:END -->" in src)


def test_generated_block_is_replaced_not_appended():
    """inject() must overwrite the previous block, or every regeneration would
    stack another copy underneath the last."""
    from compare import BEGIN, END, inject
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "R.md")
        open(p, "w").write(f"top\n{BEGIN}\nold numbers\n{END}\nbottom\n")
        out = inject(p, "new numbers")
        check("old block gone", "old numbers" not in out, out)
        check("new block present", "new numbers" in out)
        check("surrounding text kept", out.startswith("top") and out.endswith("bottom\n"))
        check("exactly one block", out.count(BEGIN) == 1 and out.count(END) == 1)


def test_inject_reports_missing_markers():
    from compare import inject
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "R.md")
        open(p, "w").write("no markers here\n")
        check("returns None rather than guessing", inject(p, "x") is None)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
