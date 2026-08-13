"""Build the four isolated MoGe export variants in interleaved order.

This is deliberately separate from ``run.py`` and ``reports/bench``. It calls
``tune_build.py`` once per graph and round, so every build gets a distinct
engine, fingerprint, timing cache, and JSON result.
"""

import argparse
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(ROOT, "models", "moge_2", "onnx")
VARIANTS = {
    "dynamo_raw": "moge-2_vits_normal_388x518_ab_dynamo.onnx",
    "dynamo_sim": "moge-2_vits_normal_388x518_ab_dynamo_sim.onnx",
    "legacy_raw": "moge-2_vits_normal_388x518_ab_legacy.onnx",
    "legacy_sim": "moge-2_vits_normal_388x518_ab_legacy_sim.onnx",
}
ORDERS = [
    ["dynamo_raw", "dynamo_sim", "legacy_raw", "legacy_sim"],
    ["legacy_sim", "legacy_raw", "dynamo_sim", "dynamo_raw"],
    ["dynamo_sim", "legacy_raw", "dynamo_raw", "legacy_sim"],
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "reports", "tune", "moge_ab"))
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    missing = [name for name in VARIANTS.values()
               if not os.path.isfile(os.path.join(ONNX_DIR, name))]
    if missing:
        ap.error("missing A/B graphs: " + ", ".join(missing))
    os.makedirs(args.out_dir, exist_ok=True)

    for round_number, order in enumerate(ORDERS, 1):
        for variant in order:
            tag = f"ab_{variant}_r{round_number}"
            print(f"\n=== START {tag} ===", flush=True)
            subprocess.run([
                sys.executable, "-X", "utf8", "tune_build.py", "moge_2",
                "--onnx", os.path.join(ONNX_DIR, VARIANTS[variant]),
                "--tag", tag,
                "--out", os.path.join(args.out_dir, f"{tag}.json"),
                "--iterations", str(args.iterations),
                "--warmup", str(args.warmup),
            ], cwd=ROOT, check=True)
            print(f"=== DONE {tag} ===", flush=True)
    print("\n=== ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
