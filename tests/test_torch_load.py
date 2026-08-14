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

# torch.hub.load is NOT one of them, and matching it was a defect in this file.
# It loads an *entrypoint* out of a repository's hubconf.py rather than a
# checkpoint, it has no weights_only parameter, and anything passed under that
# name would be forwarded to the entrypoint function as a keyword it does not
# take. Demanding weights_only there is demanding a TypeError.
#
# It is a real exposure, just a different one -- see
# test_torch_hub_load_is_reported_rather_than_conflated, which names the sites
# instead of leaving them out of the file.
NOT_A_CHECKPOINT_LOADER = {("hub", "load")}

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
    parts, root = [fn.attr], fn.value
    while isinstance(root, ast.Attribute):
        parts.append(root.attr)
        root = root.value
    if not (isinstance(root, ast.Name) and root.id == "torch"):
        return False
    return tuple(reversed(parts)) not in NOT_A_CHECKPOINT_LOADER


def _hub_load_sites():
    """torch.hub.load(...) call sites: file, line, and whether the repo is local."""
    out = []
    for path in _sources():
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            fn = getattr(node, "func", None)
            if not isinstance(node, ast.Call) or not isinstance(fn, ast.Attribute):
                continue
            if fn.attr != "load" or not isinstance(fn.value, ast.Attribute) \
                    or fn.value.attr != "hub":
                continue
            root = fn.value.value
            if not (isinstance(root, ast.Name) and root.id == "torch"):
                continue
            local = any(k.arg == "source" and isinstance(k.value, ast.Constant)
                        and k.value.value == "local" for k in node.keywords)
            out.append((os.path.relpath(path, ROOT), node.lineno, local))
    return sorted(out)


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


def test_torch_hub_load_is_reported_rather_than_conflated():
    """torch.hub.load runs a repository's hubconf.py. Name the sites.

    This check exists because the file used to fold these into
    test_no_bare_torch_load, which reported models/tr2m/onnx_export.py:140 and
    :142 as bare torch.load. They are not: weights_only does not exist on
    torch.hub.load, so that report could never be acted on, only ignored. It was
    ignored for free, because nothing in this file asserted.

    Excluding them without saying so would be the worse fix. The exposure is
    real -- `source="local"` reads a checked-out clone, and the fallback clones
    and executes a GitHub repository -- so the sites are enumerated here and a
    new one has to be added deliberately.
    """
    # Reviewed 2026-08-14. tr2m loads DINOv2 through torch.hub: once from a
    # checked-out clone under models/tr2m/.../torchhub, and once from GitHub if
    # that directory is absent. Both are upstream's own way of getting the
    # encoder, and this runs only in tr2m's own export environment.
    REVIEWED = {os.path.join("models", "tr2m", "onnx_export.py"): 2}

    sites = _hub_load_sites()
    by_file = {}
    for path, _, _ in sites:
        by_file[path] = by_file.get(path, 0) + 1
    check("torch.hub.load sites are the reviewed ones", by_file == REVIEWED,
          f"found {by_file}, reviewed {REVIEWED}")

    # A site that can only reach GitHub has no offline path at all. Every file
    # with a remote call must also carry a source="local" one.
    remote_files = {p for p, _, local in sites if not local}
    local_files = {p for p, _, local in sites if local}
    check("every file with a remote torch.hub.load also has a local branch",
          remote_files <= local_files,
          f"remote-only: {sorted(remote_files - local_files)}")


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
