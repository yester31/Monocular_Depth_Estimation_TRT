"""Catch names used but never defined, without importing anything.

These scripts cannot be imported on a machine without the model packages and a
GPU, so a plain NameError hides until the script runs -- and by then it has
often already spent minutes exporting an ONNX or building an engine. That is
exactly how `profile` survived in Uni_Depth_V2/onnx2trt.py after the profile
switch was reverted: the assignment went, a print using it stayed, and the
failure appeared only on the desktop mid-batch.

This is a deliberately small scope check, not a linter. It walks each
module-level function and flags a load of a name that is not a parameter, not
assigned anywhere in that function, not a module-level or imported name, not a
builtin, and not a comprehension or except-clause variable. Anything it is
unsure about it stays quiet on -- a false alarm here would be worse than a
miss, since it would train the reader to ignore it.
"""

import ast
import builtins
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


def _bound_names(node):
    """Every name the node binds: assignment, import, for, with, except, etc."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            out |= set(n.names)
    return out


def _star_exports(module, _seen=None):
    """Public top-level names of a repo module, without importing it.

    `from common import *` is used by every onnx2trt.py, and common.py imports
    tensorrt at module scope, so this cannot go through a real import on a
    machine with no TensorRT. Parsing gets the names, which is all that is
    needed. Returns None if the module is not one of ours.
    """
    _seen = _seen or set()
    if module in _seen:
        return set()
    _seen.add(module)
    path = os.path.join(ROOT, module.replace(".", os.sep) + ".py")
    if not os.path.isfile(path):
        return None
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = set()

    def scan(body):
        """Module scope, including inside try/if/with.

        common_runtime.py imports the CUDA bindings inside a try that falls
        back to a different module path, so a scan of top-level statements
        alone would miss `cuda` and `cudart` and report them as undefined
        wherever `from common import *` brings them in.
        """
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)          # do not descend: that is inner scope
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                names.update(_bound_names(n))
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    if a.name == "*":
                        deeper = _star_exports(getattr(n, "module", "") or "", _seen)
                        if deeper is None:
                            raise _Unknowable()
                        names.update(deeper)
                    else:
                        names.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, ast.Try):
                scan(n.body)
                for h in n.handlers:
                    scan(h.body)
                scan(n.orelse)
                scan(n.finalbody)
            elif isinstance(n, (ast.If, ast.With, ast.For, ast.While)):
                scan(n.body)
                scan(getattr(n, "orelse", []))

    try:
        scan(tree.body)
    except _Unknowable:
        return None
    return {n for n in names if not n.startswith("_")}


class _Unknowable(Exception):
    """A star import from a package we cannot read."""


def undefined_in(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    module_level = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name == "*":
                    exported = _star_exports(n.module or "")
                    if exported is None:
                        # Star import from a third-party package: the namespace
                        # is genuinely unknowable here, so judge nothing rather
                        # than report a name that does exist.
                        return None
                    module_level |= exported
                    continue
                module_level.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_level.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For,
                            ast.With, ast.If, ast.Try)):
            module_level |= _bound_names(n)

    bad = []
    for fn in [n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        local = _bound_names(fn)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id not in local and n.id not in module_level:
                    bad.append((n.id, n.lineno))
    return bad


def test_no_undefined_names():
    skipped = []
    for d in sorted(os.listdir(ROOT)):
        sub = os.path.join(ROOT, d)
        if not os.path.isdir(sub):
            continue
        for f in sorted(os.listdir(sub)):
            if not f.endswith(".py"):
                continue
            p = os.path.join(sub, f)
            bad = undefined_in(p)
            if bad is None:
                skipped.append(f"{d}/{f}")
                continue
            check(f"{d}/{f}", not bad,
                  ", ".join(f"{n} (line {l})" for n, l in bad))
    if skipped:
        print(f"  ..    {len(skipped)} skipped (star import hides the namespace)")


def test_the_case_that_motivated_this():
    """A synthetic file with exactly the Uni_Depth_V2 shape must be caught."""
    import tempfile
    src = ("def main():\n"
           "    input_h = 518\n"
           "    print(f'{profile} -> {input_h}')\n")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sample.py")
        open(p, "w").write(src)
        bad = undefined_in(p)
        check("detects the reverted-variable case",
              bad is not None and any(n == "profile" for n, _ in bad), str(bad))


def test_no_false_alarm_on_normal_code():
    import tempfile
    src = ("import os\n"
           "CONST = 3\n"
           "def helper(a, b=2):\n"
           "    total = a + b + CONST\n"
           "    items = [x * 2 for x in range(total)]\n"
           "    try:\n"
           "        return os.path.join(*[str(i) for i in items])\n"
           "    except OSError as e:\n"
           "        return str(e)\n")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sample.py")
        open(p, "w").write(src)
        check("quiet on comprehensions, defaults, except-as, builtins",
              undefined_in(p) == [], str(undefined_in(p)))


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
