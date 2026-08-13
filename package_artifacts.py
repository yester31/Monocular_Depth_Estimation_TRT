"""Assemble artifacts/<model>/<profile>-<precision>/ for the built engines.

    python package_artifacts.py                # every model with a record
    python package_artifacts.py unik3d vggt
    python package_artifacts.py --no-copy      # manifests only, no engine copy
    python package_artifacts.py --verify       # check what is already there

Run this where the engines are, i.e. on the machine that built them. On any
other machine the engine paths in reports/bench point at files that are not
here, and --no-copy still writes a truthful manifest saying so.

Why a manifest at all: an engine records none of what a caller needs -- input
layout, resolution, output meanings, the TensorRT that built it, the caveats
that make its numbers mean something. Those facts exist in spec.json and the
benchmark record; this is where they travel with the file.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import artifact, spec as spec_mod  # noqa: E402
from core.bench import load_all as load_bench  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def onnx_beside(engine_path):
    """The ONNX an engine was built from, by the naming convention."""
    if not engine_path:
        return None
    name = os.path.basename(engine_path)
    for suffix in ("_fp16.engine", "_fp32.engine", ".engine"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    folder = os.path.dirname(os.path.dirname(engine_path))
    for cand in (os.path.join(folder, "onnx", f"{name}.onnx"),
                 os.path.join(folder, "onnx", name, f"{name}.onnx")):
        if os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*")
    ap.add_argument("--no-copy", action="store_true",
                    help="write manifests without copying the engine")
    ap.add_argument("--verify", action="store_true",
                    help="check existing artifacts instead of writing")
    ap.add_argument("--out", default=artifact.ARTIFACTS)
    args = ap.parse_args()

    specs = spec_mod.load_all()
    runs = {r["model"]: r for r in load_bench(os.path.join(ROOT, "reports", "bench"))}
    wanted = set(args.models) or set(specs)

    if args.verify:
        bad = 0
        for name in sorted(wanted):
            if name not in specs:
                continue
            d = os.path.join(args.out, name, artifact.profile_id(specs[name], runs.get(name)))
            if not os.path.isdir(d):
                print(f"{name:20} not packaged")
                continue
            problems = artifact.verify(d)
            print(f"{name:20} {'ok' if not problems else '; '.join(problems)}")
            bad += bool(problems)
        return 1 if bad else 0

    made = 0
    for name in sorted(wanted):
        if name not in specs:
            print(f"{name:20} no spec")
            continue
        run = runs.get(name)
        if not run:
            print(f"{name:20} not measured - build it first")
            continue
        engine = run.get("engine_path") or ""
        onnx = onnx_beside(engine)
        d = artifact.write(specs[name], run, engine, onnx, out_root=args.out,
                           copy_engine=not args.no_copy)
        here = os.path.isfile(engine)
        size = ""
        p = os.path.join(d, "model.engine")
        if os.path.isfile(p):
            size = f"{os.path.getsize(p) / 1e6:.0f} MB"
        print(f"{name:20} {os.path.relpath(d, ROOT)}  "
              f"{size or ('engine not on this machine' if not here else 'manifest only')}")
        made += 1
    print(f"\n{made} artifact(s) under {os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
