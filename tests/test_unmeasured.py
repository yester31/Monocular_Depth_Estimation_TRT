"""A missing measurement has to be a record, and the record has to be printed.

Execution rule 9 kept failures out of the bin. It did not cover the other way a
result disappears, which is never producing one: `reports/accuracy.json` held
twelve records, `reports/profile/` thirteen files and `reports/gt/` eleven
scored models, while `PLAN.md` said 14/14. Nothing was deleted and nothing was
wrong in any single file -- the claim was only checkable by listing three
directories and counting, and nobody did until a person did it by hand.

So there are two things to hold, and they are different tests:

  1. **Every model is accounted for in every result set.** A model with no
     number must have a record saying why, or the count is a claim nobody
     checked. `test_every_model_is_accounted_for_*`.
  2. **The generated tables say so.** A record that no report prints is the
     same silence with an extra file in it. `test_the_published_*` reads the
     published markdown, not the renderer, so a generator that quietly drops
     the records fails here even if it still passes its own byte-for-byte
     consistency test.

Rule 11: these were run against deliberately broken inputs before being
committed -- a record removed, a reason code renamed, the `not measured` cell
put back to `-`, a generator's section call deleted -- and each one turns the
matching test red. A test that cannot fail is not evidence.

No GPU and no engines: everything here reads committed JSON and committed
markdown.
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from core import unmeasured  # noqa: E402

REPORTS = os.path.join(ROOT, "reports")
ACCURACY = os.path.join(REPORTS, "accuracy.json")
PROFILE_DIR = os.path.join(REPORTS, "profile")
GT_DIR = os.path.join(REPORTS, "gt")

# The result sets this file holds to account for every model, and where the
# per-model record lives in each. Adding a fourth means adding it here.
SETS = ("accuracy", "profile", "gt")


def _models():
    """The fourteen, from models/*/spec.json.

    An independent source from any of the three result sets, which is the
    point: counting the records against each other would agree with itself
    however many were missing.
    """
    from core import spec

    names = set(spec.load_all())
    if len(names) < 2:
        pytest.skip("models/*/spec.json is not on this machine")
    return names


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _records(which):
    """Every record in one result set, scored and unscored alike."""
    if which == "accuracy":
        if not os.path.exists(ACCURACY):
            pytest.skip("reports/accuracy.json is not on this machine")
        return _load(ACCURACY)
    out = []
    for stem, rec in _files(which):
        # <model>.json only. <model>_fp32.json and friends are experiments on
        # engines that were never published, and one of those standing in for
        # the published measurement is the mistake load_saved() already avoids.
        if stem == rec.get("model"):
            out.append(rec)
    return out


# Suffixes that name a variant experiment rather than the published run. A file
# carrying one of these is allowed to disagree with its own stem; anything else
# is not.
VARIANT_SUFFIXES = ("_fp32", "_uint8", "_scale_only")


def _files(which):
    """(stem, record) for every JSON in one directory, nothing filtered out."""
    d = PROFILE_DIR if which == "profile" else GT_DIR
    if not os.path.isdir(d):
        pytest.skip(f"{d} is not on this machine")
    out = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        rec = _load(os.path.join(d, name))
        if isinstance(rec, dict):
            out.append((name[:-5], rec))
    return out


# --------------------------------------------------------------------------
# 1. Every model is accounted for
# --------------------------------------------------------------------------

@pytest.mark.parametrize("which", SETS)
def test_every_model_is_accounted_for(which):
    """A number, or a record saying why there is none. No third option.

    This is the test that was missing when PLAN.md turned 12/14 into 14/14.
    Removing any one of the five not-measured records turns it red, and so
    does adding a fifteenth model without measuring it.
    """
    accounted = {r.get("model") for r in _records(which)}
    absent = sorted(_models() - accounted)
    assert not absent, (
        f"{which}: {', '.join(absent)} have neither a result nor a "
        f"not-measured record. A model missing from a result set is "
        f"indistinguishable from one that was never in the repository.")


@pytest.mark.parametrize("which", SETS)
def test_no_record_names_a_model_that_does_not_exist(which):
    """The other direction: a record for a model nobody ships is a typo.

    A misspelt model name would satisfy the count above while leaving the real
    model unaccounted for, so both directions have to be checked.
    """
    unknown = sorted({r.get("model") for r in _records(which)} - _models())
    assert not unknown, f"{which}: no models/<name>/spec.json for {unknown}"


@pytest.mark.parametrize("which", ("profile", "gt"))
def test_a_result_file_is_named_after_the_model_it_is_about(which):
    """Because the readers key on that, and disagreement is invisible.

    `core.profile_store.load_saved` and `evaluate_gt._results_for` both decide
    what a file describes from its name, and a file whose `model` field says
    something else is dropped without a word -- which is how a record written
    to close a gap can leave the gap open. Variant experiments
    (`<model>_fp32.json` and friends) are the declared exception.
    """
    for stem, rec in _files(which):
        if stem.endswith(VARIANT_SUFFIXES):
            continue
        assert rec.get("model") == stem, (
            f"reports/{which}/{stem}.json says model={rec.get('model')!r}; "
            f"every reader here finds it by its filename and will skip it")


# --------------------------------------------------------------------------
# 2. The records themselves
# --------------------------------------------------------------------------

@pytest.mark.parametrize("which", SETS)
def test_each_not_measured_record_states_which_of_the_three_it_is(which):
    """`reason_code` is the field that makes the gap actionable.

    "No number" covers an afternoon of GPU time, a question that is closed,
    and a contract nobody has built. Collapsing the three into one blank is
    what let `zipdepth` (run it) and `vggt` (do not) read the same.
    """
    found = [r for r in _records(which) if unmeasured.is_unmeasured(r)]
    for r in found:
        assert r.get("reason_code") in unmeasured.REASON_CODES, (
            f"{which}/{r.get('model')}: reason_code is "
            f"{r.get('reason_code')!r}, not one of "
            f"{sorted(unmeasured.REASON_CODES)}")
        # The code is for a machine; the sentence is what a reader gets.
        assert len(r.get("reason") or "") > 40, (
            f"{which}/{r.get('model')}: the reason is a code word, not a "
            f"reason -- {r.get('reason')!r}")


@pytest.mark.parametrize("which", SETS)
def test_a_not_measured_record_carries_no_scores(which):
    """It must not be possible to read one as a measurement.

    Every table generator sorts and formats on these keys. A record that
    carried `abs_rel: 0.0` would sort to the top of reports/gt.md as the best
    model in the repository, which is the failure this whole mechanism exists
    to avoid -- an absence rendered as a good number.
    """
    forbidden = ("outputs", "worst_rel", "verdict", "abs_rel", "rmse", "log10",
                 "delta1", "delta2", "delta3", "images", "alignment",
                 "summary", "layers", "stats")
    for r in _records(which):
        if not unmeasured.is_unmeasured(r):
            continue
        carried = sorted(k for k in forbidden if k in r)
        assert not carried, (
            f"{which}/{r.get('model')}: a not-measured record carries "
            f"{carried}, which a generator will read as a result")


def test_the_reason_codes_are_the_three_and_are_all_in_use():
    """A fourth code, or a code nobody uses, means the taxonomy drifted."""
    assert set(unmeasured.REASON_CODES) == {
        "not_run", "unsupported", "different_contract"}
    used = {r.get("reason_code")
            for which in SETS for r in _records(which)
            if unmeasured.is_unmeasured(r)}
    assert used == set(unmeasured.REASON_CODES), (
        f"in use: {sorted(used)}. A code with no record behind it is a "
        f"category nobody found, and one in use that is not declared is a "
        f"category nobody named.")


def test_record_refuses_a_reason_code_it_does_not_know():
    with pytest.raises(ValueError):
        unmeasured.record("x", "dunno", "because")


def test_split_loses_nothing_and_keeps_the_order():
    a = {"model": "a", "abs_rel": 0.1}
    b = unmeasured.record("b", "not_run", "x" * 50)
    c = {"model": "c", "abs_rel": 0.2}
    measured, missing = unmeasured.split([a, b, c])
    assert measured == [a, c]
    assert missing == [b]


def test_a_record_without_the_status_field_is_not_swept_up():
    """`is_unmeasured` keys on `status` and on nothing else.

    A result file that is merely damaged -- missing the field a caller wanted
    -- must not be reclassified as "never measured". That would turn a real
    failure into a tidy line in the Not measured section.
    """
    assert not unmeasured.is_unmeasured({"model": "a"})
    assert not unmeasured.is_unmeasured({"model": "a", "abs_rel": None})
    assert not unmeasured.is_unmeasured({"model": "a", "status": "ok"})
    assert not unmeasured.is_unmeasured(None)
    assert unmeasured.is_unmeasured({"model": "a", "status": unmeasured.STATUS})


# --------------------------------------------------------------------------
# 3. The published tables say so
# --------------------------------------------------------------------------

PUBLISHED = {
    "accuracy": os.path.join(REPORTS, "accuracy.md"),
    "profile": os.path.join(REPORTS, "comparison.md"),
    "gt": os.path.join(REPORTS, "gt.md"),
}


def _published(which):
    p = PUBLISHED[which]
    if not os.path.exists(p):
        pytest.skip(f"{os.path.relpath(p, ROOT)} is not on this machine")
    with open(p, encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("which", SETS)
def test_the_published_table_names_every_model_it_did_not_measure(which):
    """The anti-silence test, read off the artefact rather than the renderer.

    tests/test_scale_only.py and tests/test_accuracy_report.py compare a
    generator with the file it wrote, so a generator that drops these records
    and a file that does not mention them agree with each other perfectly.
    This one has nothing to agree with: the model name has to be in the
    published markdown.
    """
    text = _published(which)
    for r in _records(which):
        if not unmeasured.is_unmeasured(r):
            continue
        assert f"`{r['model']}`" in text, (
            f"{os.path.basename(PUBLISHED[which])} never mentions "
            f"{r['model']}, which has no measurement. A reader counting rows "
            f"cannot tell it apart from a model that is not in the repository.")
        assert r["reason_code"] in text, (
            f"{os.path.basename(PUBLISHED[which])} names {r['model']} but not "
            f"why it has no number")


def test_the_comparison_table_marks_the_cell_rather_than_leaving_a_dash():
    """`-` in the fp32/moved columns is not a statement.

    Those two columns are a join onto reports/profile/, and before this the
    only thing distinguishing "no profile exists" from "the profile is of a
    different engine" was a footnote nobody wrote. A row whose profile is a
    recorded absence says so in the cell.
    """
    text = _published("profile")
    missing = {r["model"] for r in _records("profile")
               if unmeasured.is_unmeasured(r)}
    assert missing, "no not-measured profile records to check"
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        model = cells[0].strip("`") if cells else ""
        if model in missing:
            assert unmeasured.CELL in cells, (
                f"the {model} row still reads {cells!r}; a blank cell in a "
                f"table of measurements reads as a small number")


def test_the_gt_table_does_not_publish_the_missing_models_as_rows():
    """Accounted for is not the same as scored.

    The reason `vggt` has no row is that the metric table cannot score a
    normalised output -- putting it in the table with a dash would be a claim
    that it was tried. It belongs in the section below the table and nowhere
    else.
    """
    text = _published("gt")
    body, _, tail = text.partition("## Not measured")
    assert tail, "reports/gt.md has no Not measured section"
    for r in _records("gt"):
        if not unmeasured.is_unmeasured(r):
            continue
        assert f"| `{r['model']}`" not in body, (
            f"{r['model']} has a row in the scored table")
        assert f"`{r['model']}`" in tail


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
