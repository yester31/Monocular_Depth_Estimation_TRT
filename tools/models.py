"""What is here, what is built, what is measured.

    python models.py                 # one line per model
    python models.py unik3d          # everything known about one
    python models.py --stale         # only what needs rebuilding or measuring

Reads spec.json and reports/, and imports nothing from a model directory, so
it runs anywhere -- including a laptop with no CUDA. That is the point of the
specs: the answer to "what does this repo contain" should not require twelve
conda environments.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import spec as spec_mod  # noqa: E402
from core.bench import load_all as load_bench  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _runs():
    return {r["model"]: r for r in load_bench(os.path.join(ROOT, "reports", "bench"))}


def _engine_of(spec, run):
    """The engine a run recorded, and whether it is still on disk."""
    p = (run or {}).get("engine_path") or ""
    return p, bool(p) and os.path.isfile(p)


def summarise(specs, runs):
    rows = []
    for name, s in sorted(specs.items()):
        run = runs.get(name)
        h, w = spec_mod.size_of(s)
        t = s["build_targets"][0]
        _, have = _engine_of(s, run)
        stats = (run or {}).get("stats") or {}
        rows.append({
            "model": name,
            "size": f"{h}x{w}",
            "profile": t.get("profile", "?"),
            "precision": t.get("precision", "?"),
            "engine": "yes" if have else ("path only" if run else "no"),
            "mean_ms": f"{stats['mean_ms']:.2f}" if stats else "-",
            "caveats": len(spec_mod.caveats(s)),
        })
    return rows


def render_table(rows):
    head = ["model", "size", "profile", "prec", "engine", "mean ms", "caveats"]
    keys = ["model", "size", "profile", "precision", "engine", "mean_ms", "caveats"]
    cells = [[str(r[k]) for k in keys] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) if cells else len(h)
              for i, h in enumerate(head)]
    out = ["  ".join(h.ljust(w) for h, w in zip(head, widths)),
           "  ".join("-" * w for w in widths)]
    for c in cells:
        out.append("  ".join(v.ljust(w) for v, w in zip(c, widths)))
    return "\n".join(out)


def render_one(name, spec, run):
    L = [f"{name}", "=" * len(name), ""]
    inp = spec["input"]
    h, w = spec_mod.size_of(spec)
    L += [f"environment   {spec['env']}",
          f"input         {inp['name']}  rank {inp['rank']}  "
          f"{inp['layout']} {inp['dtype']}  {h}x{w}",
          "outputs"]
    for o in spec["outputs"]:
        L.append(f"              {o['name']:<26} {o['meaning']}")
    L.append("profiles")
    for p, prof in spec["profiles"].items():
        s = prof["size"]
        L.append(f"              {p:<26} {s[0]}x{s[1]}")
    L.append("build targets")
    for t in spec["build_targets"]:
        L.append(f"              {t.get('profile')} / {t.get('precision')}")

    cav = spec_mod.caveats(spec)
    if cav:
        L += ["", "caveats"]
        L += [f"  - {c}" for c in cav]

    if run:
        st = run.get("stats") or {}
        v = run.get("versions") or {}
        L += ["", "last measurement",
              f"  {st.get('mean_ms', '?')} ms mean, p50 {st.get('p50_ms', '?')}, "
              f"{st.get('fps', '?')} fps",
              f"  {run.get('device', '?')}, TensorRT {v.get('tensorrt', '?')}, "
              f"{run.get('timestamp', '?')}"]
        path, have = _engine_of(spec, run)
        L.append(f"  engine {'present' if have else 'not on this machine'}: {path}")
    else:
        L += ["", "not measured yet"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="show one model in full")
    ap.add_argument("--stale", action="store_true",
                    help="only models with no measurement or no engine present")
    args = ap.parse_args()

    specs = spec_mod.load_all()
    if not specs:
        print("no specs under models/")
        return 1
    runs = _runs()

    if args.model:
        if args.model not in specs:
            print(f"unknown model {args.model!r}. known: {', '.join(sorted(specs))}")
            return 1
        print(render_one(args.model, specs[args.model], runs.get(args.model)))
        return 0

    rows = summarise(specs, runs)
    if args.stale:
        rows = [r for r in rows if r["engine"] != "yes" or r["mean_ms"] == "-"]
        if not rows:
            print("every model has an engine and a measurement")
            return 0
    print(render_table(rows))

    n_meas = sum(1 for r in summarise(specs, runs) if r["mean_ms"] != "-")
    print(f"\n{len(specs)} models, {n_meas} measured. "
          f"Speeds: reports/comparison.md. Accuracy: reports/accuracy.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
