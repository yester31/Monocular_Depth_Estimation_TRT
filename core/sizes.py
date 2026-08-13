"""Reading and rewriting the input size a model's export script pins.

A model's input size lives in its onnx_export.py as a literal, and a size sweep
has to change it without editing the file everyone else uses. The substitution
is pure text, so it lives here where it can be tested without TensorRT, a GPU,
or even OpenCV -- which is what the rest of size_sweep.py needs.

Getting this wrong is quiet in the worst way: a pattern that fails to match
would export the model at its original size, produce a second identical engine,
and the sweep would report that the input size makes no difference.
"""

import re

# The spellings in use across the models. A new one has to be added here rather
# than discovered at run time -- tests/test_sizes.py runs every export script in
# the repository through this list and fails if one of them is not covered.
PATTERNS = (
    # depth_anything_ac picks its size from the profile. A sweep pins it.
    (r"input_h, input_w = \([^)]*\)[^\n]*", "input_h, input_w = ({h}, {w})"),
    (r"input_h, input_w = \d+, \d+", "input_h, input_w = {h}, {w}"),
    (r"input_h = \d+[^\n]*\n(\s*)input_w = \d+[^\n]*",
     "input_h = {h}\n\\g<1>input_w = {w}"),
)

# Models whose input size is not a literal in their export script, so no text
# substitution can change it. depth_pro reads img_size off the loaded model --
# 1536 is upstream's, not a choice this repository makes -- and sweeping it
# would mean overriding the model's own architecture rather than its input.
NOT_SWEEPABLE = {
    "depth_pro": "img_size comes from the loaded model, not from the script",
}


def parse(text):
    """'672x896' -> (672, 896). Raises ValueError on anything else."""
    m = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", text)
    if not m:
        raise ValueError(f"expected HxW, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def rewrite(source, h, w):
    """`source` with its input size replaced, or None if no pattern matched.

    None rather than the unchanged source: a caller that cannot tell the two
    apart would export at the original size and call it a measurement.
    """
    for pattern, replacement in PATTERNS:
        new, n = re.subn(pattern, replacement.format(h=h, w=w), source, count=1)
        if n:
            return new
    return None


def sweepable(model):
    """(True, None) or (False, why). Stated here rather than discovered."""
    why = NOT_SWEEPABLE.get(model)
    return (why is None), why


def find(source):
    """The (h, w) a script currently pins, or None."""
    m = re.search(r"input_h, input_w = (\d+), (\d+)", source)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"input_h = (\d+)[^\n]*\n\s*input_w = (\d+)", source)
    return (int(m.group(1)), int(m.group(2))) if m else None
