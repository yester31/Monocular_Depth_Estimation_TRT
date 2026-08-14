"""A record for a measurement that does not exist, and why it does not.

Execution rule 9 says a failure, an OOM or an unsupported case is not deleted
from the results. It said nothing about a measurement that was **never taken**,
so those lived in prose -- and prose is not counted. `PLAN.md` claimed 12/14
had become 14/14 while `reports/accuracy.json` held twelve records and
`reports/profile/` thirteen files; the only way to see it was to list the
directories by hand, which is what eventually happened.

So the absence is written down in the same place the presence would have gone:
a record in `reports/accuracy.json`, a `reports/profile/<model>.json`, a
`reports/gt/<model>.json`. One record, one missing measurement.

    {
      "model": "tr2m",
      "status": "not_measured",
      "reason_code": "different_contract",
      "reason": "one sentence, in the report's own voice",
      "see": "docs/model_contracts.md - tr2m 평가 계약",
      "recorded": "2026-08-14"
    }

**Old readers do not break, by construction.** `status` is a field they have
never seen, and every field they *do* read -- `outputs`, `abs_rel`, `summary`
-- is simply absent, which is the same situation as a record written before
that field existed. The repository's existing convention is that a reader
tolerates a missing field (`o.setdefault(...)` in verify_accuracy, the
`engine_bytes` fallback in build_conditions), and these records rely on
exactly that and add nothing new. What a reader must not do is drop them
silently: `split()` is here so that separating them from the scored records
and *reporting* them is one call rather than two decisions.

Three reason codes, because "no number" hides three different situations and
each calls for a different action:

    not_run             The measurement is defined, the harness can express it,
                        and nothing is blocking it. It needs GPU time nobody
                        has spent yet. -> run it.
    unsupported         The measurement is not defined for this model. Running
                        it anyway would produce a number that means nothing.
                        -> do not run it; measure the other thing instead.
    different_contract  The measurement is defined but this repository cannot
                        currently express it -- a second engine input, a
                        per-image prompt, a manifest that does not exist yet.
                        -> build the contract first, then run it.

The distinction is the whole point. "12 of 14" says nothing about whether the
two gaps are an afternoon of GPU time or a design question.
"""

STATUS = "not_measured"

# code -> what a reader should conclude, and what the next action is. The text
# travels into the generated reports, so it is written for that reader.
REASON_CODES = {
    "not_run": "not run yet - defined and unblocked, needs GPU time",
    "unsupported": "not defined for this model - a number here would mean nothing",
    "different_contract": "needs an evaluation contract this repository does not have yet",
}

# What a table cell says where a number would have gone. Short enough not to
# widen a column past readability, explicit enough that it cannot be read as a
# measurement of zero or as a rounding of something small.
CELL = "not measured"


def record(model, reason_code, reason, see=None, recorded=None, **extra):
    """Build one record. `reason` is a sentence, not a code word.

    The code is for a machine and the sentence is for a person, and neither
    substitutes for the other: `unsupported` alone does not say which contract
    is being violated, and a paragraph alone cannot be counted.
    """
    if reason_code not in REASON_CODES:
        raise ValueError(
            f"unknown reason_code {reason_code!r}; "
            f"expected one of {sorted(REASON_CODES)}")
    out = {"model": model, "status": STATUS, "reason_code": reason_code,
           "reason": reason}
    if see:
        out["see"] = see
    if recorded:
        out["recorded"] = recorded
    out.update(extra)
    return out


def is_unmeasured(rec):
    """True for a record that carries no measurement.

    Deliberately keyed on `status` alone. A record that merely lacks the field
    a caller wanted is not therefore an unmeasured one -- it may predate that
    field -- and conflating the two would let a genuinely broken result file
    disappear into this category.
    """
    return isinstance(rec, dict) and rec.get("status") == STATUS


def split(records):
    """(measured, unmeasured), in the order given.

    Every generator that renders a table calls this instead of filtering by
    hand, so that the second half has to be dealt with somewhere. The failure
    mode being designed out is a list comprehension that keeps the scored rows
    and never mentions the rest.
    """
    measured, missing = [], []
    for r in records:
        (missing if is_unmeasured(r) else measured).append(r)
    return measured, missing


def describe(rec):
    """One line about one record, for a bullet list in a generated report."""
    bits = [rec.get("reason") or REASON_CODES.get(rec.get("reason_code"), "")]
    if rec.get("see"):
        bits.append(f"See {rec['see']}.")
    return " ".join(b for b in bits if b)


def section(records, heading="Not measured", intro=None):
    """Markdown lines for the unmeasured records, or [] when there are none.

    Grouped by reason code with the code's own explanation above the group,
    because the three are not the same news: one is a queue, one is a closed
    question, one is unbuilt machinery.
    """
    records = [r for r in records if is_unmeasured(r)]
    if not records:
        return []
    L = [f"## {heading}", "",
         intro or ("These have no result file. They are recorded here rather "
                   "than left out of the tables above, so that the count of "
                   "rows is not mistaken for the count of models."),
         ""]
    for code in REASON_CODES:
        group = sorted((r for r in records if r.get("reason_code") == code),
                       key=lambda r: r.get("model", ""))
        if not group:
            continue
        L += [f"**{code}** — {REASON_CODES[code]}", ""]
        L += [f"- `{r.get('model', '?')}` — {describe(r)}" for r in group]
        L.append("")
    unknown = sorted((r for r in records
                      if r.get("reason_code") not in REASON_CODES),
                     key=lambda r: r.get("model", ""))
    if unknown:
        # A record written by something that did not use record(). Say so
        # rather than filing it under a code it did not claim.
        L += ["**reason not stated**", ""]
        L += [f"- `{r.get('model', '?')}` — {describe(r) or 'no reason recorded'}"
              for r in unknown]
        L.append("")
    return L
