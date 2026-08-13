"""Rewriting the input size in a model's export script.

The failure this guards against is silent. If a pattern stops matching, the
sweep exports the model at its original size, builds a second identical engine,
and reports that input size makes no difference to accuracy -- a conclusion
that looks like a finding.

So the last test here runs the rewriter against every real onnx_export.py in
the repository rather than against invented strings.
"""

import glob
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sizes import NOT_SWEEPABLE, find, parse, rewrite, sweepable  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_parse_accepts_the_form_the_command_line_uses():
    assert parse("672x896") == (672, 896)
    assert parse(" 518 x 518 ") == (518, 518)


@pytest.mark.parametrize("bad", ["672", "672*896", "x896", "", "672x"])
def test_parse_refuses_anything_else(bad):
    with pytest.raises(ValueError):
        parse(bad)


def test_rewrites_the_single_line_spelling():
    src = "    input_h, input_w = 434, 560\n    dynamo = True\n"
    out = rewrite(src, 672, 896)
    assert "input_h, input_w = 672, 896" in out
    assert "dynamo = True" in out


def test_rewrites_the_two_line_spelling_and_keeps_indentation():
    src = "    input_h = 518 # 1036\n    input_w = 518 # 1386\n    x = 1\n"
    out = rewrite(src, 672, 896)
    assert "input_h = 672" in out and "input_w = 896" in out
    # The second line has to stay indented, or the rewritten script will not
    # parse and the failure will look like the model's fault.
    assert "\n    input_w = 896" in out
    assert "x = 1" in out


def test_only_the_first_occurrence_is_touched():
    src = ("    input_h, input_w = 518, 518\n"
           "    # input_h, input_w = 700, 700\n")
    out = rewrite(src, 672, 896)
    assert out.count("672") == 1
    assert "700, 700" in out


def test_no_match_returns_none_rather_than_the_original():
    # The whole point: a caller must be able to tell "not rewritten" from
    # "rewritten", because exporting at the original size would produce a
    # sweep that says size does not matter.
    assert rewrite("nothing to see here", 1, 2) is None


def test_find_reads_back_what_rewrite_wrote():
    for src in ("    input_h, input_w = 434, 560\n",
                "    input_h = 518 # a\n    input_w = 518 # b\n"):
        assert find(rewrite(src, 672, 896)) == (672, 896)


def test_rewrites_the_conditional_spelling():
    # depth_anything_ac picks its size from the profile; a sweep pins it.
    src = "    input_h, input_w = (518, 518) if profile == 'bench' else (518, 700)\n"
    out = rewrite(src, 672, 896)
    assert out.strip() == "input_h, input_w = (672, 896)"


def test_every_export_script_is_either_rewritable_or_declared_not():
    scripts = sorted(glob.glob(os.path.join(ROOT, "models", "*", "onnx_export.py")))
    assert scripts, "no export scripts found; the glob is wrong"
    unmatched = []
    for path in scripts:
        model = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8") as f:
            src = f.read()
        can, _ = sweepable(model)
        if can and rewrite(src, 672, 896) is None:
            unmatched.append(model)
    assert not unmatched, (
        f"core/sizes.py cannot rewrite the input size of: {', '.join(unmatched)}. "
        f"Add the spelling to PATTERNS, or declare the model in NOT_SWEEPABLE "
        f"with the reason -- a sweep that cannot rewrite a script must not fall "
        f"back to exporting it unchanged.")


def test_a_declared_exception_carries_a_reason():
    for model, why in NOT_SWEEPABLE.items():
        assert why and len(why) > 10, f"{model} is excluded without saying why"
        assert sweepable(model) == (False, why)
