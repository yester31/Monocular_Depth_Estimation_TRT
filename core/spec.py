"""Read and validate models/<name>/spec.json.

A spec is a **dependency-free declaration**: it can be read in the shared
`trte` environment with nothing but the standard library. That is the whole
point of it existing. The alternative -- importing the model's own python to
learn its input size -- drags torch and the upstream package into the shared
runtime and collapses the environment boundary the repository is built on.

So this module imports json and os, and nothing else, forever.

What a spec is not: a second source of truth. Every field here is either
checked against the scripts by tests/test_spec.py or was taken from a
measurement in reports/. A spec that drifts from the code is worse than no
spec, because it reads as authoritative.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

SCHEMA = 1

# Fields every spec must carry. Kept short on purpose: a required field that
# cannot be checked is a field that will quietly go stale.
REQUIRED = ("schema", "name", "env", "input", "outputs", "profiles",
            "build_targets")


class SpecError(ValueError):
    pass


def path_for(name):
    return os.path.join(MODELS, name, "spec.json")


def load(name_or_path):
    """Load one spec by model name or by path."""
    p = (name_or_path if name_or_path.endswith(".json")
         else path_for(name_or_path))
    if not os.path.isfile(p):
        raise SpecError(f"no spec at {p}")
    with open(p, encoding="utf-8") as f:
        spec = json.load(f)
    validate(spec, p)
    return spec


def load_all():
    """Every spec under models/, keyed by name."""
    out = {}
    if not os.path.isdir(MODELS):
        return out
    for d in sorted(os.listdir(MODELS)):
        p = path_for(d)
        if os.path.isfile(p):
            out[d] = load(p)
    return out


def validate(spec, where="<spec>"):
    """Structural check. Agreement with the scripts is tests/test_spec.py."""
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise SpecError(f"{where}: missing {missing}")
    if spec["schema"] != SCHEMA:
        raise SpecError(f"{where}: schema {spec['schema']}, expected {SCHEMA}")

    inp = spec["input"]
    for k in ("name", "rank", "dtype", "layout"):
        if k not in inp:
            raise SpecError(f"{where}: input.{k} missing")
    if inp["rank"] not in (4, 5):
        raise SpecError(f"{where}: input.rank {inp['rank']} is not 4 or 5")

    if not spec["outputs"]:
        raise SpecError(f"{where}: outputs is empty")
    for o in spec["outputs"]:
        for k in ("name", "meaning"):
            if k not in o:
                raise SpecError(f"{where}: output.{k} missing in {o}")

    if not spec["profiles"]:
        raise SpecError(f"{where}: profiles is empty")
    for pname, prof in spec["profiles"].items():
        size = prof.get("size")
        if not (isinstance(size, list) and len(size) == 2
                and all(isinstance(v, int) and v > 0 for v in size)):
            raise SpecError(f"{where}: profile {pname} size must be [h, w], got {size}")

    if not spec["build_targets"]:
        raise SpecError(f"{where}: build_targets is empty — "
                        "a spec that builds nothing is a spec nobody runs")
    for t in spec["build_targets"]:
        if t.get("profile") not in spec["profiles"]:
            raise SpecError(f"{where}: build target {t} names an unknown profile")
    return spec


def size_of(spec, profile=None):
    """The [h, w] a profile builds at. Defaults to the first build target."""
    if profile is None:
        profile = spec["build_targets"][0]["profile"]
    return tuple(spec["profiles"][profile]["size"])


def caveats(spec):
    """Conditions that must travel with this model's numbers, if any."""
    return list(spec.get("caveats", []))
