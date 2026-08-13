"""Nothing printed to the console may contain a character cp949 cannot encode.

The console here is Korean Windows. print() raises UnicodeEncodeError on an em
dash, and the traceback replaces whatever the line was going to say -- so the
failure lands exactly where someone was trying to read an explanation. It has
happened three times: torch.onnx's check mark during export (documented in the
README), an em dash in common.py's "Rebuilding engine" message, which hid why
an engine was being rebuilt, and a separator in run.py's stage header.

PYTHONUTF8=1 fixes it for a process that remembers to set it. This does not
depend on remembering.

Only strings inside a print() call are checked. Comments and docstrings are
full of em dashes and are never encoded to the console.
"""

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {"later", ".git", "__pycache__", "tests"}

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


def _strings_in(node):
    """Every literal string under a node, including f-string fragments."""
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            yield n.value


def _printed(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        return
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "print"):
            for arg in n.args:
                for s in _strings_in(arg):
                    yield n.lineno, s


def test_printed_strings_are_ascii():
    bad, seen = [], 0
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(base, f)
            for lineno, s in _printed(path):
                seen += 1
                offenders = sorted({c for c in s if ord(c) > 127})
                if offenders:
                    # ascii(), not repr(): repr puts the character itself in
                    # the message, and printing the report then dies of the
                    # very thing it is reporting. It did.
                    bad.append(f"{os.path.relpath(path, ROOT)}:{lineno}  "
                               f"{' '.join(ascii(c) for c in offenders)}")

    check("found print calls", seen >= 100, f"only {seen}")
    check("all printable on cp949", not bad,
          "\n        " + "\n        ".join(bad))


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
