"""One timing loop and one result format for every model.

Each onnx2trt.py grew its own copy of the same loop, and each README carries
numbers typed in by hand. That makes the models impossible to compare: the
loops disagree on what they measure, and nothing records the conditions a
number was taken under, so two figures a month apart may differ only because
the driver changed.

This module fixes the loop in one place and writes a machine-readable result
next to it. `compare.py` turns those files into reports/comparison.md, so no
number is ever transcribed by hand again.

Three things worth knowing about the numbers this produces:

**Wall clock, not kernel time.** The measured region is one full
`do_inference()` -- host-to-device copy, kernels, device-to-host copy -- and
ends after `cuda.synchronize()`. That is what a caller actually waits for.
CUDA-event timing would report a smaller number by excluding the copies; it is
not what this measures.

**Percentiles, because the mean hides stalls.** The old loops accumulated a
sum and divided. Keeping every sample costs nothing and shows when a model is
merely slow versus occasionally very slow.

**Never divide FPS by pixel count.** Measured on depth_anything_ac: 518x518
takes 14.4 ms and 700x518 takes 24.5 ms -- 1.70x the time for 1.35x the
pixels. Attention is quadratic in token count, so a per-pixel figure flatters
large inputs and is meaningless across models. Compare at equal input size or
state the size beside the number.
"""

import json
import math
import os
import platform
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

import numpy as np

SCHEMA = 1


@dataclass
class Bench:
    """Timing samples plus everything needed to reproduce them."""

    model: str
    samples_ms: List[float] = field(default_factory=list)
    warmup: int = 0

    # what was run
    backend: str = "tensorrt"
    precision: str = "fp32"
    profile: str = "bench"          # bench | native
    variant: str = "single"         # single | split, for VGGT-style layouts
    input_h: int = 0
    input_w: int = 0

    # where it was run
    device: str = ""
    driver: str = ""
    versions: Dict[str, str] = field(default_factory=dict)
    host: str = field(default_factory=platform.node)

    # what it produced, so a speed change with an output change is visible
    outputs: Dict[str, dict] = field(default_factory=dict)
    engine_path: str = ""
    onnx_sha256: str = ""
    timestamp: str = ""
    notes: str = ""

    @property
    def iterations(self) -> int:
        return len(self.samples_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms) if self.samples_ms else 0.0

    @property
    def fps(self) -> float:
        m = self.mean_ms
        return 1000.0 / m if m else 0.0

    def pct(self, q: float) -> float:
        """q in [0,100]. Nearest-rank, so the value is a real sample.

        rank = ceil(q/100 * N), which puts p99 of 1..100 at 99 -- not the
        maximum. Rounding instead of ceiling gets this wrong, because Python
        rounds halves to even and 99.5 becomes 100.
        """
        if not self.samples_ms:
            return 0.0
        s = sorted(self.samples_ms)
        i = min(len(s) - 1, max(0, math.ceil(q / 100.0 * len(s)) - 1))
        return s[i]

    def stats(self) -> dict:
        if not self.samples_ms:
            return {}
        return {
            "iterations": self.iterations,
            "warmup": self.warmup,
            "mean_ms": round(self.mean_ms, 4),
            "min_ms": round(min(self.samples_ms), 4),
            "p50_ms": round(self.pct(50), 4),
            "p90_ms": round(self.pct(90), 4),
            "p99_ms": round(self.pct(99), 4),
            "max_ms": round(max(self.samples_ms), 4),
            "stdev_ms": round(statistics.stdev(self.samples_ms), 4)
            if self.iterations > 1 else 0.0,
            "fps": round(self.fps, 2),
        }

    def report(self) -> str:
        """The lines the old loops printed, plus the spread they omitted."""
        if not self.samples_ms:
            return "[MDET] no samples"
        s = self.stats()
        total = sum(self.samples_ms) / 1000.0
        return "\n".join([
            f"[MDET] {self.iterations} iterations time: {total:.4f} [sec]",
            f"[MDET] Average FPS: {s['fps']:.2f} [fps]",
            f"[MDET] Average inference time: {s['mean_ms']:.2f} [msec]",
            f"[MDET] p50 {s['p50_ms']:.2f} / p90 {s['p90_ms']:.2f} / "
            f"p99 {s['p99_ms']:.2f} [msec], min {s['min_ms']:.2f}, "
            f"stdev {s['stdev_ms']:.2f}",
        ])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema"] = SCHEMA
        d["stats"] = self.stats()
        # samples are kept, but after stats so the summary reads first
        d["samples_ms"] = [round(x, 4) for x in self.samples_ms]
        return d


def measure(fn: Callable[[], object], *, warmup: int = 10, iterations: int = 100,
            sync: Optional[Callable[[], None]] = None):
    """Run fn and return (last return value, per-iteration milliseconds).

    `sync` runs inside the timed region, after fn, and must block until the
    work is really finished -- normally torch.cuda.synchronize. Without it the
    loop times the enqueue, not the inference, and reports a fantasy number.
    """
    if sync is None:
        try:
            import torch
            sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
        except ImportError:
            sync = lambda: None

    for _ in range(warmup):
        fn()
    sync()

    out, samples = None, []
    for _ in range(iterations):
        # perf_counter, not time.time: monotonic and the highest resolution
        # the platform offers. time.time can step backwards on a clock
        # adjustment, which silently corrupts a long run.
        t0 = time.perf_counter()
        out = fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return out, samples


def summarize_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Enough of each output to notice if a speed change moved the result.

    NaN and Inf are counted rather than folded into the statistics; moge_2 and
    metric_anything mark invalid pixels with inf, and one such pixel would
    otherwise swallow min/max.
    """
    out = {}
    for name, arr in outputs.items():
        a = np.asarray(arr)
        finite = np.isfinite(a)
        v = a[finite]
        out[name] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "finite": int(finite.sum()),
            "nonfinite": int(a.size - finite.sum()),
            "min": float(v.min()) if v.size else None,
            "max": float(v.max()) if v.size else None,
            "mean": float(v.mean()) if v.size else None,
        }
    return out


def collect_env() -> dict:
    """Versions and GPU. A benchmark without these cannot be reproduced."""
    env, device, driver = {}, "", ""
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda or ""
    except ImportError:
        pass
    try:
        import tensorrt as trt
        env["tensorrt"] = trt.__version__
    except ImportError:
        pass
    try:
        import subprocess
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=10).strip().splitlines()[0]
    except Exception:
        pass
    env["python"] = platform.python_version()
    return {"versions": env, "device": device, "driver": driver}


def save(bench: Bench, out_dir: str) -> str:
    """Write reports/bench/<model>_<profile>_<variant>_<precision>.json.

    The name carries the axes that make two runs incomparable, so a new
    precision or profile lands beside the old one instead of overwriting it.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not bench.timestamp:
        bench.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not bench.device:
        bench.__dict__.update(collect_env())

    parts = [bench.model, bench.profile, bench.variant, bench.precision]
    if bench.input_h and bench.input_w:
        parts.insert(1, f"{bench.input_h}x{bench.input_w}")
    path = os.path.join(out_dir, "_".join(str(p) for p in parts if p) + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bench.to_dict(), f, indent=2, ensure_ascii=False)
    return path


REPORTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "bench")


def record(model: str, samples_ms, *, outputs: Optional[Dict[str, np.ndarray]] = None,
           out_dir: Optional[str] = None, echo: bool = True, **kw) -> Bench:
    """Print the usual [MDET] lines and write the result file. One call per script.

    Every onnx2trt.py ends with this, so adding a field to the record format
    reaches all models at once instead of twelve near-identical edits.
    """
    b = Bench(model=model, samples_ms=list(samples_ms), **kw)
    if outputs:
        b.outputs = summarize_outputs(outputs)
    if echo:
        print(b.report())
    path = save(b, out_dir or REPORTS)
    if echo:
        try:
            shown = os.path.relpath(path)
        except ValueError:      # different drive on Windows
            shown = path
        print(f"[MDET] result -> {shown}")
    return b


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {d.get('schema')}, expected {SCHEMA}")
    return d


def load_all(bench_dir: str) -> List[dict]:
    if not os.path.isdir(bench_dir):
        return []
    out = []
    for name in sorted(os.listdir(bench_dir)):
        if name.endswith(".json"):
            out.append(load(os.path.join(bench_dir, name)))
    return out
