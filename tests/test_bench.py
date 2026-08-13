"""Tests for core/bench.py — the shared timing loop and result format.

A benchmark module that is itself wrong is worse than none, because the
numbers look authoritative. These check the arithmetic against hand-computed
values rather than against the implementation.
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.bench import (  # noqa: E402
    SCHEMA, Bench, load, load_all, measure, measure_staged, save,
    summarize_outputs,
)

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


def test_stats_arithmetic():
    b = Bench(model="demo", samples_ms=[10.0, 20.0, 30.0, 40.0])
    check("iterations", b.iterations == 4)
    check("mean", abs(b.mean_ms - 25.0) < 1e-9, str(b.mean_ms))
    # 1000 / 25 ms = 40 fps
    check("fps from mean", abs(b.fps - 40.0) < 1e-9, str(b.fps))
    s = b.stats()
    check("min", s["min_ms"] == 10.0)
    check("max", s["max_ms"] == 40.0)


def test_percentiles_are_real_samples():
    """Nearest-rank, so every percentile is a value that actually occurred.

    Interpolating would invent a latency no iteration ever took.
    """
    b = Bench(model="demo", samples_ms=[float(x) for x in range(1, 101)])
    s = sorted(b.samples_ms)
    for q in (50, 90, 99):
        check(f"p{q} is an observed sample", b.pct(q) in s, str(b.pct(q)))
    check("p50 of 1..100", b.pct(50) == 50.0, str(b.pct(50)))
    check("p99 of 1..100", b.pct(99) == 99.0, str(b.pct(99)))
    check("p100 clamps to max", b.pct(100) == 100.0, str(b.pct(100)))
    check("p0 clamps to min", b.pct(0) == 1.0, str(b.pct(0)))


def test_percentiles_survive_one_sample():
    b = Bench(model="demo", samples_ms=[7.0])
    check("single sample p90", b.pct(90) == 7.0)
    check("stdev needs two", b.stats()["stdev_ms"] == 0.0)


def test_empty_is_not_a_crash():
    b = Bench(model="demo")
    check("empty stats", b.stats() == {})
    check("empty fps", b.fps == 0.0)
    check("empty report", "no samples" in b.report())


def test_measure_counts_and_warms_up():
    calls = []

    def fn():
        calls.append(len(calls))
        return "result"

    out, samples = measure(fn, warmup=3, iterations=5, sync=lambda: None)
    check("warmup excluded from samples", len(samples) == 5, str(len(samples)))
    check("warmup still ran", len(calls) == 8, str(len(calls)))
    check("returns last value", out == "result")
    check("samples are positive", all(s >= 0 for s in samples))


def test_measure_syncs_inside_the_timed_region():
    """If sync ran outside, its cost would vanish from the samples and the
    reported latency would be the enqueue time, not the inference time."""
    order = []
    _, samples = measure(lambda: order.append("run"), warmup=0, iterations=2,
                         sync=lambda: order.append("sync"))
    # The leading sync drains whatever the warmup queued, so the first timed
    # iteration does not absorb it. With warmup=0 there is nothing to drain,
    # but the call still happens and belongs in the expectation.
    check("drains before timing, then syncs per run",
          order == ["sync", "run", "sync", "run", "sync"], str(order))
    check("two samples", len(samples) == 2)


def test_summarize_outputs_isolates_nonfinite():
    """moge_2 and metric_anything mark invalid pixels with inf. Folding one
    into min/max would make the whole summary useless."""
    a = np.array([[1.0, 2.0], [np.inf, np.nan]], np.float32)
    s = summarize_outputs({"depth": a})["depth"]
    check("counts finite", s["finite"] == 2, str(s["finite"]))
    check("counts nonfinite", s["nonfinite"] == 2, str(s["nonfinite"]))
    check("max ignores inf", s["max"] == 2.0, str(s["max"]))
    check("mean ignores nan", abs(s["mean"] - 1.5) < 1e-6, str(s["mean"]))
    check("shape recorded", s["shape"] == [2, 2])


def test_summarize_all_nonfinite():
    s = summarize_outputs({"d": np.array([np.inf, np.nan])})["d"]
    check("no finite values", s["min"] is None and s["mean"] is None)


def test_save_load_roundtrip():
    b = Bench(model="depth_anything_v2", samples_ms=[5.0, 6.0, 7.0],
              precision="fp16", profile="bench", input_h=518, input_w=518,
              device="RTX 3080", warmup=10)
    with tempfile.TemporaryDirectory() as td:
        p = save(b, td)
        check("filename carries the axes",
              os.path.basename(p) ==
              "depth_anything_v2_518x518_bench_single_fp16.json",
              os.path.basename(p))
        d = load(p)
        check("schema stamped", d["schema"] == SCHEMA)
        check("stats embedded", d["stats"]["mean_ms"] == 6.0)
        check("samples kept", d["samples_ms"] == [5.0, 6.0, 7.0])
        check("timestamp filled", bool(d["timestamp"]))
        check("load_all finds it", len(load_all(td)) == 1)


def test_differing_configs_do_not_overwrite():
    """fp16 and fp32 of the same model must coexist, or the comparison table
    silently shows whichever ran last."""
    with tempfile.TemporaryDirectory() as td:
        for prec in ("fp16", "fp32"):
            save(Bench(model="m", samples_ms=[1.0], precision=prec,
                       input_h=518, input_w=518, device="x"), td)
        check("both files kept", len(load_all(td)) == 2,
              str(sorted(os.listdir(td))))


def test_schema_mismatch_is_loud():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bad.json")
        json.dump({"schema": SCHEMA + 99}, open(p, "w"))
        try:
            load(p)
            check("rejects unknown schema", False, "no error raised")
        except ValueError:
            check("rejects unknown schema", True)


def test_report_mentions_spread():
    b = Bench(model="m", samples_ms=[10.0, 10.0, 40.0])
    txt = b.report()
    check("report has fps", "fps" in txt, txt)
    check("report has percentiles", "p99" in txt, txt)


def test_stage_breakdown_is_absent_unless_measured():
    """A missing breakdown must stay missing. Filling it with zeros would read
    as "the copies are free", which is the opposite of what is unknown."""
    b = Bench(model="m", samples_ms=[10.0, 10.0])
    s = b.stats()
    check("no h2d key", "h2d_ms" not in s, str(sorted(s)))
    check("no overhead key", "host_overhead_ms" not in s, str(sorted(s)))
    check("report stays quiet", "compute" not in b.report())


def test_stage_breakdown_names_what_it_does_not_cover():
    """GPU phases will not sum to the wall clock. The remainder is launch
    overhead and the synchronize, and on a fast model it is most of the time,
    so it is reported rather than left for the reader to subtract."""
    b = Bench(model="m", samples_ms=[10.0, 10.0],
              stage_samples_ms={"h2d_ms": [1.0, 1.0], "compute_ms": [4.0, 4.0],
                                "d2h_ms": [1.0, 1.0]})
    s = b.stats()
    check("h2d averaged", s["h2d_ms"] == 1.0, str(s.get("h2d_ms")))
    check("compute averaged", s["compute_ms"] == 4.0, str(s.get("compute_ms")))
    check("remainder named", s["host_overhead_ms"] == 4.0,
          str(s.get("host_overhead_ms")))
    check("report shows the split", "compute" in b.report(), b.report())


def test_stage_remainder_never_goes_negative():
    """Event timing and perf_counter measure different things and can disagree
    by a hair; a negative "overhead" would be nonsense in a published table."""
    b = Bench(model="m", samples_ms=[5.0],
              stage_samples_ms={"h2d_ms": [1.0], "compute_ms": [4.5],
                                "d2h_ms": [1.0]})
    check("clamped at zero", b.stats()["host_overhead_ms"] == 0.0,
          str(b.stats()["host_overhead_ms"]))


def test_measure_staged_collects_one_probe_per_iteration():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return calls["n"]

    def probe():
        return {"h2d_ms": 0.5, "compute_ms": 2.0, "d2h_ms": 0.25}

    out, samples, stages = measure_staged(fn, probe, warmup=2, iterations=5,
                                          sync=lambda: None)
    check("warmup not sampled", len(samples) == 5, str(len(samples)))
    check("fn ran warmup+iterations", calls["n"] == 7, str(calls["n"]))
    check("one probe per sample", len(stages["compute_ms"]) == 5,
          str(len(stages["compute_ms"])))
    check("last value returned", out == 7, str(out))


def test_measure_staged_tolerates_a_silent_probe():
    """A run configured without a timer should degrade to plain measure(),
    not raise partway through a hundred iterations."""
    _, samples, stages = measure_staged(lambda: None, lambda: None,
                                        warmup=0, iterations=3, sync=lambda: None)
    check("samples still collected", len(samples) == 3)
    check("no stages invented", stages == {}, str(stages))


def test_stage_samples_survive_a_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        b = Bench(model="m", samples_ms=[10.0], precision="fp16",
                  stage_samples_ms={"h2d_ms": [1.0], "compute_ms": [4.0],
                                    "d2h_ms": [1.0]})
        d = load(save(b, td))
        check("samples kept", d["stage_samples_ms"]["compute_ms"] == [4.0],
              str(d.get("stage_samples_ms")))
        check("stats kept", d["stats"]["compute_ms"] == 4.0)


def test_old_records_still_load():
    """Twelve records were written before the breakdown existed. Adding an
    optional field must not strand them."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "old.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"schema": SCHEMA, "model": "m", "samples_ms": [1.0],
                       "stats": {"mean_ms": 1.0}}, f)
        d = load(path)
        check("loads", d["model"] == "m")
        check("no breakdown claimed", "h2d_ms" not in d["stats"])


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
