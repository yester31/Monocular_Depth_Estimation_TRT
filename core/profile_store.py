"""Reading and writing reports/profile/, with no TensorRT in sight.

Split out of profile.py because that module subclasses trt.IProfiler and so
cannot be imported without TensorRT -- while compare.py, which wants these
profiles to put an engine's fp32 and data-movement share next to its speed,
runs on a laptop that has none. The join silently disabled itself there: the
columns printed "-" for every row and looked like missing data rather than a
missing import.
"""

import json
import os

REPORTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "profile")


def save(model, payload, out_dir=None):
    out_dir = out_dir or REPORTS
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_saved(out_dir=None):
    """model -> the saved profile, for reports that want to sit beside a speed.

    Only the plain per-model files. An experiment writes <model>_<tag>.json and
    those describe engines that were never published, so joining them to a
    benchmark row would attach a number to the wrong thing.
    """
    out_dir = out_dir or REPORTS
    found = {}
    if not os.path.isdir(out_dir):
        return found
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(out_dir, name), encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            continue
        model = payload.get("model")
        if model and name == f"{model}.json":
            found[model] = payload
    return found
