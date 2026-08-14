"""Turn reports/bench/*.json into reports/comparison.md.

Every number in the generated table comes from a file written by core/bench.py
during an actual run. Nothing here is typed by hand, which is the whole point:
the READMEs currently carry figures whose conditions nobody recorded, so two
of them cannot be compared even in principle.

The table is grouped by input size rather than flattened across all models.
That is deliberate. Three models do not run at 518 -- depth_pro is fixed at
1536, metric3d_v2 at 616x1064, moge_2 at 388x518 -- and attention cost is
quadratic in token count, so a single ranked list would just be a list of
which model got the smallest input. See docs/model_contracts.md 2.

usage:
    python compare.py                 # reports/bench -> reports/comparison.md
    python compare.py --check         # exit 1 if the file is out of date
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import unmeasured  # noqa: E402
from core.bench import load_all  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "reports", "bench")
OUT = os.path.join(ROOT, "reports", "comparison.md")

# What each model's depth numbers actually mean. Mirrors the classification in
# docs/model_contracts.md 1; keep the two in step. This lives here rather than
# in a data file until spec.json lands in Phase 4.
KIND = {
    # depth_anything_v2 ships both a relative and a metric checkpoint and the
    # scripts default to metric_model=True with the hypersim (indoor) weights,
    # so that is what a default build measures. Flip metric_model in
    # onnx_export.py and onnx2trt.py together for the relative one.
    "depth_anything_v2": "metric (hypersim) by default",
    "depth_anything_ac": "relative",
    # DA3Metric-Large is the Metric series (upstream README): the depth
    # output is metric. This said "relative" until the clone was read.
    "depth_anything_v3": "metric + sky mask",
    "distill_any_depth": "relative",
    "depth_pro": "metric",
    "metric3d_v2": "canonical (not metres - D12)",
    "moge_2": "point map + metric_scale",
    "metric_anything": "point map + metric_scale",
    "unidepth_v2": "point map + intrinsics",
    "unik3d": "point map + intrinsics",
    "vggt": "geometry (scale unknown)",
    "streamvggt": "geometry (scale unknown)",
    # Inverse depth, so a large value is near. Named that way rather than
    # "relative" because the sign convention is the opposite of the other
    # relative rows and reading it the wrong way round inverts the picture.
    "zipdepth": "relative (inverse depth)",
    # Metric, but produced by rescaling a relative model with a per-pixel
    # scale and shift predicted from a sentence -- not measured from the image
    # alone like the other metric rows.
    "tr2m": "metric (text-conditioned)",
}

# Every model directory now matches its key exactly, so there is no mapping to
# keep. FOLDER survives as an empty dict because callers still consult it and
# a future upstream rename could reintroduce a mismatch; the coverage check in
# tests/test_compare.py fails loudly if one appears undeclared.
FOLDER = {}

MODELS = os.path.join(ROOT, "models")

# Caveats that must travel with the number, or a reader will draw the wrong
# conclusion from a table that looks authoritative.
#
# Read from each model's spec.json rather than kept here as a second copy.
# They used to be duplicated, which meant the table and the spec could disagree
# about the same model and neither would look wrong on its own.
def _load_caveats():
    try:
        from core.spec import caveats, load_all
    except Exception:
        return {}
    out = {}
    for name, spec in load_all().items():
        c = caveats(spec)
        if c:
            out[name] = "; ".join(c)
    return out


CAVEAT = _load_caveats()


def _fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def _size(r):
    h, w = r.get("input_h") or 0, r.get("input_w") or 0
    return f"{h}x{w}" if h and w else "?"


def _load_profiles():
    # profile_store, not profile: the latter subclasses trt.IProfiler and does
    # not import without TensorRT, which this machine has none of. That import
    # error used to be swallowed here and every health column printed "-",
    # which reads as "not measured yet" rather than "not loaded".
    from core.profile_store import load_saved
    return load_saved()


PROFILES = _load_profiles()


def _health(run):
    """(fp32 share, data-movement share) for the engine this row measured.

    Both come from reports/profile/, written by a separate run of
    tools/profile_model.py, so the first thing to establish is that the profile is
    about this engine. distill_any_depth sat in this table at 10.02 ms with
    86.6% of its time in fp32 layers, and nothing in the table said so; the
    point of these two columns is that the next one is visible without anyone
    thinking to go and look.

    `stale` rather than a number when the engine was rebuilt after the profile
    was taken -- a plausible-looking figure from the previous engine is worse
    than a blank.
    """
    p = PROFILES.get(run["model"])
    # A recorded absence, not a missing file. `-` here has always meant "no
    # profile joined to this row" and covers three different situations at
    # once; where reports/profile/ says why there is none, say so instead.
    if unmeasured.is_unmeasured(p):
        return unmeasured.CELL, unmeasured.CELL
    if not p or p.get("variant", "single") != run.get("variant", "single"):
        return "-", "-"
    try:
        from core.build_conditions import same_engine
    except Exception:                                         # noqa: BLE001
        return "-", "-"
    match = same_engine(run, p)
    if match is None:
        return "-", "-"
    if not match:
        return "stale", "stale"
    s = p.get("summary") or {}
    def pct(key):
        v = s.get(key)
        return "-" if v is None else f"{v * 100:.1f}%"
    return pct("fp32_output_share"), pct("movement_share")


def _rows(runs):
    for r in sorted(runs, key=lambda r: (r["model"], r.get("precision", ""))):
        s = r.get("stats") or {}
        fp32, movement = _health(r)
        yield [
            f"`{r['model']}`",
            r.get("variant", "single"),
            # Four of these models offer several backbones and the choice moves
            # the time more than anything in the builder. A row without it
            # invites a comparison that is really about model size.
            r.get("encoder") or "-",
            r.get("precision", "?"),
            _fmt(s.get("mean_ms")),
            _fmt(s.get("p50_ms")),
            _fmt(s.get("p90_ms")),
            _fmt(s.get("fps")),
            fp32,
            movement,
            KIND.get(r["model"], "?"),
        ]


HEAD = ["model", "variant", "encoder", "precision", "mean ms", "p50", "p90", "fps",
        "fp32", "moved", "output"]


# Columns whose values are numbers and should sit right-aligned. Decided by
# name rather than position, so a table with a different column set still lines
# up -- the root README summary drops two of them.
NUMERIC = {"mean ms", "p50", "p90", "p99", "fps", "min", "fp32", "moved"}


def _table(rows, head=HEAD):
    widths = [max(len(str(c)) for c in [h] + [r[i] for r in rows])
              for i, h in enumerate(head)]
    align = ["---:" if h in NUMERIC else "---" for h in head]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |",
           "| " + " | ".join(align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w)
                                     for c, w in zip(r, widths)) + " |")
    return "\n".join(out)


def render(runs):
    if not runs:
        return ("# Model comparison\n\n"
                "No benchmark results yet. Run a model's `onnx2trt.py`; it "
                "writes one JSON per configuration into `reports/bench/`.\n")

    lines = ["# Model comparison", ""]
    lines.append("Generated by `compare.py` from `reports/bench/*.json`. "
                 "Do not edit by hand -- rerun the script.")
    lines.append("")

    devices = sorted({r.get("device", "") for r in runs if r.get("device")})
    trts = sorted({(r.get("versions") or {}).get("tensorrt", "") for r in runs} - {""})
    stamps = sorted(r.get("timestamp", "") for r in runs if r.get("timestamp"))
    lines.append(f"- GPU: {', '.join(devices) or 'unrecorded'}")
    lines.append(f"- TensorRT: {', '.join(trts) or 'unrecorded'}")
    if stamps:
        lines.append(f"- Measured: {stamps[0][:10]} to {stamps[-1][:10]}")
    lines.append(f"- Runs: {len(runs)}")
    lines.append("")

    if len(devices) > 1:
        lines += ["> **Mixed hardware.** The rows below were not all measured "
                  "on the same GPU, so they are not comparable. Rerun on one "
                  "machine before reading anything into the ranking.", ""]

    # Precision is in every row, but a reader scanning the ms column will not
    # notice that one model ran at a different one. Say it once, at the top.
    by_prec = defaultdict(list)
    for r in runs:
        by_prec[r.get("precision", "?")].append(r["model"])
    if len(by_prec) > 1:
        odd = sorted(by_prec, key=lambda p: len(by_prec[p]))[:-1]
        detail = "; ".join(
            f"{p}: " + ", ".join(f"`{m}`" for m in sorted(set(by_prec[p])))
            for p in odd)
        lines += [f"> **Mixed precision.** Most rows are "
                  f"{sorted(by_prec, key=lambda p: -len(by_prec[p]))[0]}, "
                  f"but {detail}. Compare within a precision, not across.", ""]

    # Group by profile, then input size. Latency across different input sizes
    # is not a like-for-like comparison.
    by_profile = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by_profile[r.get("profile", "bench")][_size(r)].append(r)

    for profile in sorted(by_profile):
        lines.append(f"## Profile `{profile}`")
        lines.append("")
        sizes = by_profile[profile]
        for size in sorted(sizes, key=lambda s: -len(sizes[s])):
            group = sizes[size]
            lines.append(f"### Input {size}"
                         + ("" if len(group) > 1 else "  (single model)"))
            lines.append("")
            lines.append(_table(list(_rows(group))))
            lines.append("")
            # Per-run notes: which encoder, which checkpoint. Two runs of the
            # same model with different weights are different measurements and
            # the table has to say so.
            noted = [(r["model"], r["notes"]) for r in group if r.get("notes")]
            if noted:
                lines += [f"- `{m}` -- {n}" for m, n in sorted(set(noted))]
                lines.append("")

        if len(sizes) > 1:
            lines += [
                "Latency is **not** comparable across the sections above. "
                "Attention cost grows faster than pixel count -- measured on "
                "`depth_anything_ac`, 1.35x the pixels cost 1.70x the time -- "
                "so dividing by resolution does not rescue the comparison.",
                "",
            ]

    present = {r["model"] for r in runs}
    notes = [(m, c) for m, c in sorted(CAVEAT.items()) if m in present]
    if notes:
        lines += ["## Caveats", "",
                  "These numbers are valid as latency. The *output* carries a "
                  "condition recorded in `docs/model_contracts.md`.", ""]
        lines += [f"- `{m}` -- {c}" for m, c in notes]
        lines.append("")

    # Why the `fp32` and `moved` columns are blank for some rows. Those two are
    # a join onto reports/profile/, and a row whose profile does not exist
    # prints the same thing as a row whose profile did not match -- so the
    # reason has to be stated somewhere other than the cell.
    lines += unmeasured.section(
        [p for m, p in sorted(PROFILES.items()) if m in present],
        heading="Layer profiles not measured",
        intro=f"The `fp32` and `moved` columns read `{unmeasured.CELL}` for "
              f"these rows. The speed beside them was measured; the layer "
              f"profile behind those two columns was not.")

    missing = sorted(set(KIND) - present)
    if missing:
        lines += ["## Not measured", "",
                  "No result file exists for these yet, so they are absent "
                  "from the tables above rather than slow:", "",
                  ", ".join(f"`{m}`" for m in missing), ""]

    return "\n".join(lines) + "\n"


BEGIN, END = "<!-- BENCH:BEGIN -->", "<!-- BENCH:END -->"


def inject(path, body):
    """Replace the text between the markers. Returns the new file contents.

    The markers exist so the READMEs can carry measured numbers without anyone
    retyping them. A README that quotes a figure by hand is stale from the
    next run onward, and this repository already had twelve of those.
    """
    src = open(path, encoding="utf-8").read()
    i, j = src.find(BEGIN), src.find(END)
    if i == -1 or j == -1:
        return None
    return src[:i] + BEGIN + "\n" + body.rstrip() + "\n" + src[j:]


def render_summary(runs):
    """The root README block: one row per model, grouped by input size."""
    if not runs:
        return "_No measurements yet. Run a model's `onnx2trt.py`._"

    dev = sorted({r.get("device", "") for r in runs if r.get("device")})
    trt = sorted({(r.get("versions") or {}).get("tensorrt", "") for r in runs} - {""})
    lines = [f"Measured on {', '.join(dev) or 'an unrecorded GPU'}, "
             f"TensorRT {', '.join(trt) or 'unrecorded'}, on `data/example.jpg`.",
             "Generated by `compare.py` — do not edit between the markers.", ""]

    by_size = defaultdict(list)
    for r in runs:
        by_size[_size(r)].append(r)

    for size in sorted(by_size, key=lambda s: (-len(by_size[s]), s)):
        group = sorted(by_size[size], key=lambda r: (r.get("stats") or {}).get("mean_ms", 0))
        lines.append(f"**Input {size}**"
                     + ("" if len(group) > 1 else "  (only model at this size)"))
        lines.append("")
        rows = []
        for r in group:
            s = r.get("stats") or {}
            rows.append([f"`{r['model']}`", r.get("precision", "?"),
                         _fmt(s.get("mean_ms")), _fmt(s.get("p50_ms")),
                         _fmt(s.get("fps")), KIND.get(r["model"], "?")])
        lines.append(_table(rows, ["model", "precision", "mean ms", "p50", "fps", "output"]))
        lines.append("")

    if len(by_size) > 1:
        lines += ["Speeds do **not** compare across those groups: attention cost "
                  "grows faster than pixel count, so a model at a smaller input "
                  "is not therefore faster. Full table with p90/p99 and the "
                  "per-model caveats: [reports/comparison.md](reports/comparison.md).",
                  ""]
    return "\n".join(lines)


def _clock_note(run):
    """How the GPU clock stood, when the record says.

    Two numbers that were taken under different clock policies are not
    comparable, and -- because TensorRT chooses kernels by timing them during
    the build -- an engine built under a swinging clock is not even the same
    engine twice. Saying so beside the number is the only way a reader can
    tell which kind of measurement they are looking at.
    """
    clock, top = run.get("clock_mhz") or 0, run.get("clock_max_mhz") or 0
    if not clock or not top:
        return ""
    # Stated as the reading it is. nvidia-smi on this card no longer answers
    # whether a lock is in force -- Applications Clocks reports "deprecated" --
    # so a clock near the maximum long after the run is evidence of a lock and
    # not proof of one. The reading is recorded; the policy is in PLAN.md.
    return f", GPU clock {clock} of {top} MHz after the run"


def render_model_block(run):
    """The per-model README block."""
    s = run.get("stats") or {}
    v = run.get("versions") or {}
    return "\n".join([
        f"- Input **{_size(run)}**, precision **{run.get('precision','?')}**, "
        f"{s.get('iterations','?')} iterations after {s.get('warmup','?')} warmup",
        f"- **{_fmt(s.get('mean_ms'))} ms** mean, "
        f"p50 {_fmt(s.get('p50_ms'))} / p90 {_fmt(s.get('p90_ms'))} / "
        f"p99 {_fmt(s.get('p99_ms'))}, min {_fmt(s.get('min_ms'))}",
        f"- **{_fmt(s.get('fps'))} fps**",
        f"- {run.get('device','?')}, TensorRT {v.get('tensorrt','?')}, "
        f"torch {v.get('torch','?')}"
        + (f", driver {run['driver']}" if run.get("driver") else "")
        + _clock_note(run),
        "",
        "Generated by `compare.py` from `reports/bench/`. Timing covers one "
        "`do_inference()` — host-to-device copy, kernels, device-to-host copy — "
        "and nothing else; post-processing is outside the measured region.",
    ])


def write_readmes(runs, check=False):
    """Update the root README and each model README. Returns (changed, missing)."""
    targets = [(os.path.join(ROOT, "README.md"), render_summary(runs))]
    for r in runs:
        folder = FOLDER.get(r["model"], r["model"])
        path = os.path.join(MODELS, folder, "README.md")
        targets.append((path, render_model_block(r)))

    changed, missing = [], []
    for path, body in targets:
        if not os.path.exists(path):
            missing.append(path)
            continue
        new = inject(path, body)
        if new is None:
            missing.append(path)
            continue
        if new != open(path, encoding="utf-8").read():
            changed.append(path)
            if not check:
                open(path, "w", encoding="utf-8", newline="").write(new)
    return changed, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", default=BENCH_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if anything is out of date")
    ap.add_argument("--no-readme", action="store_true",
                    help="only regenerate reports/comparison.md")
    args = ap.parse_args()

    runs = load_all(args.bench_dir)
    text = render(runs)

    stale = []
    if args.check:
        old = (open(args.out, encoding="utf-8").read()
               if os.path.exists(args.out) else "")
        if old != text:
            stale.append(args.out)
    else:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")

    if not args.no_readme:
        changed, missing = write_readmes(runs, check=args.check)
        for p in missing:
            print(f"  no BENCH markers: {os.path.relpath(p, ROOT)}")
        if args.check:
            stale += changed
        else:
            for p in changed:
                print(f"wrote {os.path.relpath(p, ROOT)}")

    if args.check:
        if stale:
            print("out of date - run: python compare.py")
            for p in stale:
                print(f"  {os.path.relpath(p, ROOT)}")
            return 1
        print("everything is up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
