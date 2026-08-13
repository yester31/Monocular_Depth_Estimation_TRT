"""Every torch.load in this repository passes weights_only.

torch.load without it unpickles arbitrary objects: loading a checkpoint is
running whatever the file's author put in it. Every checkpoint here was probed
first (see the table below) and all six accept weights_only=True, so this is
not a trade-off between safety and working code -- it costs nothing.

The reason for a test rather than one sweep: the eight call sites live in
scripts that cannot be imported without their own conda environment and a GPU,
so a reverted or newly added bare torch.load would go unnoticed until someone
read the diff. This reads the source instead, which works anywhere.

Probed 2026-08-13, torch 2.11.0+cu128, each in its own environment:

    depth_anything_AC_vits.pth                     OK
    depth_anything_v2_vits.pth                     OK
    depth_anything_v2_metric_hypersim_vits.pth     OK
    StreamVGGT/ckpt/checkpoints.pth                OK
    unidepth-v2-vitb14/pytorch_model.bin           OK
    vggt/checkpoints/model.pt                      OK

If a future checkpoint refuses, the error names the exact global that blocks
it; allow that one with torch.serialization.add_safe_globals rather than
turning the flag off.
"""

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


# torch.hub.load_state_dict_from_url downloads and unpickles, same exposure.
LOADERS = {"load", "load_state_dict_from_url"}

SKIP = {"later", ".git", "__pycache__"}


def _sources():
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(base, f)


def _is_loader(node):
    """torch.load(...) / torch.hub.load_state_dict_from_url(...), by shape.

    Matching the attribute name rather than resolving the import keeps this
    from needing to know how each script spells its torch import, at the cost
    of also matching an unrelated obj.load() -- which is why a match must also
    look like a checkpoint load, i.e. it is only ever reported, never rewritten.
    """
    fn = node.func
    if not isinstance(fn, ast.Attribute) or fn.attr not in LOADERS:
        return False
    root = fn.value
    while isinstance(root, ast.Attribute):
        root = root.value
    return isinstance(root, ast.Name) and root.id == "torch"


def test_no_bare_torch_load():
    bare, seen = [], 0
    for path in _sources():
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_loader(node):
                continue
            seen += 1
            if not any(k.arg == "weights_only" for k in node.keywords):
                bare.append(f"{os.path.relpath(path, ROOT)}:{node.lineno}")

    check("found the load sites", seen >= 8, f"only {seen}")
    check("none bare", not bare, "\n        " + "\n        ".join(bare))


def test_none_of_them_disable_it():
    """weights_only=False is the same exposure written out longhand."""
    off = []
    for path in _sources():
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_loader(node):
                continue
            for k in node.keywords:
                if k.arg == "weights_only" and isinstance(k.value, ast.Constant) \
                        and k.value.value is False:
                    off.append(f"{os.path.relpath(path, ROOT)}:{node.lineno}")
    check("none opt out", not off, str(off))


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
