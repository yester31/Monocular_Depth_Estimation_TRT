"""core/common.py must not move sys.path[0].

Every model script puts the repo root at index 1 so that sys.path[0] stays its
own directory, and metric_anything reads sys.path[0] back to reach a vendored
copy of `moge`:

    sys.path.insert(1, os.path.join(sys.path[0], "metric_anything/models/..."))

core/common.py is imported before that line runs. When it inserted the repo root at
index 0, the join resolved against the root instead, and the script died on
ModuleNotFoundError several imports later -- pointing at a file that had
nothing to do with the change. It cost a twelve-model measurement pass.

Checked by reading the source because importing core/common.py needs TensorRT.
"""

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path_inserts(source):
    """(index, line) for every sys.path.insert(...) with a literal index."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "insert"):
            continue
        target = f.value
        if not (isinstance(target, ast.Attribute) and target.attr == "path"
                and isinstance(target.value, ast.Name) and target.value.id == "sys"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            found.append((node.args[0].value, node.lineno))
    return found


def test_common_does_not_take_over_sys_path_zero():
    with open(os.path.join(ROOT, "core", "common.py"), encoding="utf-8") as f:
        inserts = _path_inserts(f.read())
    assert not [i for i, _ in inserts if i == 0], (
        f"core/common.py inserts into sys.path at index 0 (lines "
        f"{[ln for i, ln in inserts if i == 0]}); model scripts rely on "
        f"sys.path[0] still being their own directory")


def test_the_check_can_see_an_insert():
    # Guard against the parser silently matching nothing and passing forever.
    assert _path_inserts("import sys\nsys.path.insert(0, 'x')\n") == [(0, 2)]
    assert _path_inserts("import sys\nsys.path.insert(1, 'x')\n") == [(1, 2)]
    assert _path_inserts("import sys\nsys.path.append('x')\n") == []
