"""Under what GPU conditions was an engine built, and does that invalidate it?

Kept out of common.py so it can be tested without TensorRT installed, the same
reason engine_staleness() was split out of get_engine().

The short version: TensorRT picks kernels by timing candidates on the GPU while
it builds. A card whose clock swings between idle and boost gives it noisy
timings, so the choice is partly random. Four builds of unik3d from one ONNX,
identical options, byte-identical fingerprints, produced 19.04, 13.35, 8.41 and
14.49 ms. Three builds with the clock pinned gave 9.66, 9.62 and 9.68.
"""

import os
import subprocess

# The share of the card's maximum below which a reading is plainly an idle
# clock rather than a pin. Only used for reporting, never for a decision.
IDLE_FRACTION = 0.6


def gpu_clock():
    """(current graphics clock, card maximum) in MHz, or (0, 0) if unknown."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.gr,clocks.max.gr",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=10)
        cur, top = (int(p.strip()) for p in out.strip().splitlines()[0].split(","))
        return cur, top
    except Exception:                                         # noqa: BLE001
        return 0, 0


def path_for(engine_file_path):
    return os.path.splitext(engine_file_path)[0] + ".buildinfo.json"


def stamp(engine_file_path):
    """Identify the exact engine file, so two records can be shown to describe it.

    A benchmark and a layer profile are taken by separate runs of separate
    scripts, and the profile is what says whether an engine is running most of
    its time in fp32 -- the symptom that gave distill_any_depth away. Joining
    them on the model name alone would let a profile of a replaced engine sit
    beside a fresh measurement and look like evidence about it. Size and
    mtime both change on a rebuild, and neither costs anything to read.
    """
    try:
        st = os.stat(engine_file_path)
    except OSError:
        return {}
    return {"engine_bytes": st.st_size, "engine_mtime": int(st.st_mtime)}


def same_engine(a, b):
    """True, False, or None when either record predates the stamp.

    None rather than False on purpose. A record written before stamping is not
    evidence that the engine changed, and reporting it as a mismatch would put
    "stale" beside twelve rows that are current -- which teaches the reader to
    ignore the word in the one case it matters.
    """
    keys = ("engine_bytes", "engine_mtime")
    if not all((a or {}).get(k) and (b or {}).get(k) for k in keys):
        return None
    return all(a[k] == b[k] for k in keys)


def complaint(info, now_mhz=0):
    """What is wrong with the conditions an engine was built under, or None.

    The clock is deliberately not part of the fingerprint. Hashing it would
    rebuild every engine on every run on an unpinned machine, and a check that
    expensive gets switched off. But it has to be visible somewhere, because it
    decides what is *in* the engine while the fingerprint says nothing is
    wrong -- which is the exact shape of the stale-engine defect that cost
    distill_any_depth 2.3x: a build input outside the fingerprint, and no
    signal anywhere. This is the signal.

    Pinned is read off two samples taken either side of the build rather than
    asked of the driver, which no longer answers: nvidia-smi reports
    Applications Clocks as deprecated on this card. A pin holds the same clock
    at idle and under load, so the two samples match; without one, a build of
    several minutes leaves them far apart.
    """
    if not info:
        return "built before build conditions were recorded"
    before = info.get("clock_before_mhz", 0)
    after = info.get("clock_after_mhz", 0)
    top = info.get("clock_max_mhz", 0)
    if not before or not after or not top:
        return None
    if before != after:
        return (f"built under a free GPU clock ({before} -> {after} of {top} MHz); "
                f"its kernel choice came from noisy timings")
    if now_mhz and now_mhz != after:
        return (f"built at {after} MHz, running at {now_mhz}; the engine is fine "
                f"but a speed measured now is not comparable to one measured then")
    return None
