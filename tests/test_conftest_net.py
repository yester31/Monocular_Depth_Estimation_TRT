"""The safety net in conftest.py has to be falsifiable itself.

conftest.py turns a `check()` failure that was only collected into a real pytest
failure. That is the sort of guarantee that can quietly stop working -- a pytest
release changes the hook protocol, someone renames `_failures` -- and if it does,
everything goes green again and the twelve modules that rely on it go back to
being unfailable. Nothing else in the suite would notice, because when the net
works there is nothing to see.

So this runs pytest on a throwaway module in the old shape and asserts the run
comes back non-zero, and on the same module with the check passing and asserts
it comes back zero.
"""

import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFTEST = os.path.join(ROOT, "tests", "conftest.py")

MODULE = textwrap.dedent('''
    _failures = []

    def check(name, cond, detail=""):
        print(f"  {{'PASS' if cond else 'FAIL'}}  {{name}}")
        if not cond:
            _failures.append(name)

    def test_collected_only():
        check("the thing under test", {verdict})
''')


def _run(tmp_path, verdict):
    (tmp_path / "conftest.py").write_text(
        open(CONFTEST, encoding="utf-8").read(), encoding="utf-8")
    (tmp_path / "test_sample.py").write_text(
        MODULE.format(verdict=verdict), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=str(tmp_path))


def test_a_collected_failure_fails_the_run(tmp_path):
    r = _run(tmp_path, "False")
    assert r.returncode != 0, (
        "a check() that only appended to _failures still passed pytest; the net "
        f"in tests/conftest.py is not working\n{r.stdout}\n{r.stderr}")
    assert "only collected" in r.stdout, r.stdout


def test_a_passing_check_still_passes(tmp_path):
    r = _run(tmp_path, "True")
    assert r.returncode == 0, (
        f"the net fails a module whose checks all pass\n{r.stdout}\n{r.stderr}")
