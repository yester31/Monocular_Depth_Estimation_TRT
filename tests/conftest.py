"""Make a collected check failure fail pytest.

Twelve of the test modules here predate pytest in this repository. They were
written to be run as `python tests/test_x.py`, so each one carries the same
shape:

    _failures = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}...")
        if not cond:
            _failures.append(name)

    if __name__ == "__main__":
        ...
        sys.exit(1 if _failures else 0)

Run as a script that works. Run under pytest it does not: the test function
records the failure and returns normally, pytest sees no exception, and the test
passes. Counted across the twelve files that is 240 assertions that could not
fail a `pytest tests -q`, in a repository whose merge check is `pytest tests -q`.

Rewriting 240 call sites into asserts would be a large diff for the same
guarantee this gives, and it would not stop the thirteenth file from arriving
with the same shape. So the guarantee lives here instead: after every test,
anything the module appended to `_failures` while that test ran is raised.

The modules keep their `check()` and their `__main__` runner, which is the part
that is actually useful about them -- one run reports every failing check rather
than stopping at the first.
"""

import os
import sys

# compare.py / evaluate_gt.py / models.py / verify_accuracy.py moved to tools/
# on 2026-08-14 so that the repository root answers "what do I run" with two
# files. They are still importable modules, and the tests still import them by
# bare name, so tools/ goes on the path here rather than in five test files.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest


def _failures_list(item):
    """The module-level `_failures` of the test being run, if it has one."""
    module = getattr(item, "module", None)
    failures = getattr(module, "_failures", None)
    return failures if isinstance(failures, list) else None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    failures = _failures_list(item)
    before = len(failures) if failures is not None else 0

    # If the test raises, that is already a pytest failure and the exception
    # propagates from here untouched -- there is nothing to add.
    result = yield

    if failures is None:
        return result
    new = failures[before:]
    if new:
        raise AssertionError(
            f"{len(new)} check() call{'s' if len(new) > 1 else ''} failed and "
            f"were only collected:\n  " + "\n  ".join(str(n) for n in new) +
            "\n(detail is in the captured stdout above)")
    return result
