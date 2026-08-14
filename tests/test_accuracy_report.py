"""reports/accuracy.md must be reproducible from reports/accuracy.json.

Nothing covered this file before. It carries twelve models' engine-against-ONNX
numbers and it was the only published table with no test saying where those
numbers came from -- which is the state that lets a hand-edit survive, and lets
`verify_accuracy.py --from-json` drift from the file it wrote.

It also had no input-size column, so twenty-two rows sorted by `rel` read as one
ranking while depth_pro was measured on 1536x1536 and moge_2 on 388x518.
Execution rule 8. The column is checked here against two independent sources:
the shape of the tensor both sides were actually fed, and spec.json.

No GPU, no engine: re-rendering from the committed JSON is the whole point.
"""

import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PUBLISHED = os.path.join(ROOT, "reports", "accuracy.md")
RESULTS = os.path.join(ROOT, "reports", "accuracy.json")
INPUTS = os.path.join(ROOT, "reports", "inputs")


def _artefacts():
    if not (os.path.exists(PUBLISHED) and os.path.exists(RESULTS)):
        pytest.skip("reports/accuracy.md and accuracy.json are not on this machine")
    with open(PUBLISHED, encoding="utf-8") as f:
        text = f.read()
    with open(RESULTS, encoding="utf-8") as f:
        return text, json.load(f)


def _cells(line):
    """Split a markdown row on unescaped pipes.

    `ref \\|mean\\|` is one cell whose text contains two pipes. A plain
    `line.split("|")` reads it as three, which is exactly the mistake the
    published file itself used to make -- see the header comment in
    verify_accuracy.render.
    """
    return [c.strip().replace("\\|", "|")
            for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def _rows(text):
    """The verdict rows of the published table, as dicts keyed by header."""
    lines = [l for l in text.splitlines() if l.startswith("|")]
    head = _cells(lines[0])
    out = []
    for line in lines[2:]:
        cells = _cells(line)
        if len(cells) == len(head) and cells[0] in ("ok", "WARN", "FAIL"):
            out.append(dict(zip(head, cells)))
    return out


def test_the_published_table_renders_from_the_json():
    """A hand-edit, or a renderer change nobody republished, shows up here."""
    import verify_accuracy

    text, results = _artefacts()
    assert verify_accuracy.render(results) == text


def test_every_row_states_the_input_size_it_was_measured_at():
    text, _ = _artefacts()
    rows = _rows(text)
    assert len(rows) >= 12, f"only {len(rows)} rows parsed"
    for r in rows:
        assert "input HxW" in r, f"{r['model']}: no input size column"
        assert re.fullmatch(r"\d+x\d+", r["input HxW"]), \
            f"{r['model']}: input size reads {r['input HxW']!r}"
        assert r["prec"] in ("fp16", "fp32"), r


def test_the_input_size_is_the_shape_of_the_tensor_that_was_fed():
    """Rule 7: the column comes out of the artefact, not off a keyboard.

    reports/inputs/<model>.npy is the array verify_accuracy hands to both the
    engine and the ONNX graph, so its last two dimensions are the size the row
    was measured at by construction. Cross-checked against spec.json, which is
    an independent declaration of the same fact.
    """
    from core import spec as spec_mod

    text, _ = _artefacts()
    specs = spec_mod.load_all()
    for r in _rows(text):
        model = r["model"].strip("`")
        path = os.path.join(INPUTS, f"{model}.npy")
        if not os.path.exists(path):
            pytest.skip(f"reports/inputs/{model}.npy is not on this machine")
        shape = np.load(path, mmap_mode="r").shape
        assert r["input HxW"] == f"{shape[-2]}x{shape[-1]}", (
            f"{model}: table says {r['input HxW']}, the recorded input tensor "
            f"is {shape}")
        assert model in specs, f"{model} has no spec.json"
        assert (shape[-2], shape[-1]) == tuple(spec_mod.size_of(specs[model])), (
            f"{model}: input tensor is {shape[-2]}x{shape[-1]}, spec.json says "
            f"{spec_mod.size_of(specs[model])}")


def test_the_numbers_in_the_table_are_the_numbers_in_the_json():
    """The column was added; nothing else may have moved.

    Read off the markdown and compared with the JSON, so a future column that
    shifts a value into the neighbouring cell fails here rather than being
    published.
    """
    from core import unmeasured

    text, results = _artefacts()
    by_key = {(r["model"], o["index"]): (r, o)
              for r in results
              if "skip" not in r and not unmeasured.is_unmeasured(r)
              for o in r["outputs"]}
    rows = _rows(text)
    assert len(rows) == len(by_key), f"{len(rows)} rows against {len(by_key)} outputs"
    for row in rows:
        key = (row["model"].strip("`"), int(row["out"]))
        assert key in by_key, key
        r, o = by_key[key]
        assert row["prec"] == r.get("precision", "?"), key
        assert row["rel"] == f"{o['rel_mean']:.4%}", key
        assert row["max abs"] == f"{o['max_abs']:.3g}", key
        assert row["scale"] == f"{o['scale']:.4f}", key
        assert row[""] == o["verdict"], key


def test_the_rows_are_worst_first():
    """The order is the claim the table makes about which engine to look at."""
    text, _ = _artefacts()
    rel = [float(r["rel"].rstrip("%")) for r in _rows(text)]
    assert rel == sorted(rel, reverse=True), rel


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
