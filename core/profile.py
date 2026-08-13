"""Where the time goes inside one engine.

The benchmark says a model takes 4.3 ms. It does not say what for. Every
optimisation discussion in this repository so far has been a guess about that,
and the one real speedup found -- 20% on unidepth_v2, from float64 in the graph
blocking kernel fusion -- was found by accident while checking accuracy, not by
looking. A layer profile would have shown it immediately.

Two sources, and they answer different questions:

**IProfiler** gives measured time per layer. Note that attaching it serialises
execution, so the total is larger than the benchmark's -- the *shares* are what
matter, not the absolute numbers.

**EngineInspector** gives the engine's structure: which layers TensorRT fused
into one, and what precision each ended up running at. A layer that stayed
fp32 inside an fp16 engine is a finding; so is a Transpose or a Cast surviving
as its own layer, since those should normally be folded into a neighbour.

Nothing here decides anything. It produces the list that decisions come from.
"""

import json
import os
from collections import OrderedDict

import tensorrt as trt


class LayerTimer(trt.IProfiler):
    """Accumulates per-layer time across iterations.

    TensorRT reports a layer once per execution, and the same name can appear
    more than once in one pass, so times are summed and the count is kept --
    dividing by iterations alone would be wrong for a layer that runs twice.
    """

    def __init__(self):
        trt.IProfiler.__init__(self)
        self.total_ms = OrderedDict()
        self.calls = OrderedDict()

    def report_layer_time(self, layer_name, ms):
        self.total_ms[layer_name] = self.total_ms.get(layer_name, 0.0) + ms
        self.calls[layer_name] = self.calls.get(layer_name, 0) + 1

    def rows(self, iterations):
        """[{name, ms, calls, share}] sorted slowest first."""
        total = sum(self.total_ms.values()) or 1.0
        out = [{
            "name": name,
            "ms": round(ms / max(iterations, 1), 6),
            "calls_per_iter": round(self.calls[name] / max(iterations, 1), 2),
            "share": round(ms / total, 6),
        } for name, ms in self.total_ms.items()]
        out.sort(key=lambda r: -r["ms"])
        return out


def inspect(engine):
    """Engine structure: one entry per layer, from TensorRT's own inspector.

    Requires the engine to have been built with profiling_verbosity DETAILED,
    which common.get_engine sets. Without it the layer names come back as
    opaque identifiers and nothing here is readable.
    """
    inspector = engine.create_engine_inspector()
    layers = []
    for i in range(engine.num_layers):
        raw = inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)
        try:
            layers.append(json.loads(raw))
        except json.JSONDecodeError:
            layers.append({"Name": raw})
    return layers


def _precision_of(layer):
    """What this layer actually produced.

    The inspector does not report a "Precision" field; the datatype is on each
    tensor, as `Format/Datatype` ("Float", "Half", "Int8", ...). Reading the
    first output is what tells you a layer stayed fp32 inside an fp16 engine --
    which TensorRT does silently when a layer has no fp16 kernel or when
    keeping it would lose too much range.
    """
    outs = layer.get("Outputs") or []
    if not outs:
        return "?"
    return (outs[0].get("Format/Datatype") or "?").split()[0]


# Layer types that rearrange bytes rather than compute with them. Resize is
# left out on purpose: an interpolation is real work the model asked for,
# unlike a Reformat, which exists only because two neighbours disagree about
# layout. Concat/Constant/NoOp are usually free but are counted when they are
# not, since a Concat that costs time is a copy.
MOVEMENT_TYPES = {"Reformat", "Shuffle", "Cast", "Copy", "Slice",
                  "Concatenation", "Concat", "NoOp", "Constant", "Transpose"}


def summarize(rows, layers, mean_ms=None):
    """The few numbers worth reading before the full table.

    `mean_ms` is the benchmark's figure. It is reported beside the profile's
    own total precisely because they disagree: profiling serialises the
    layers, so its total runs high. Printing both stops that gap from being
    read as a regression.
    """
    prof_total = sum(r["ms"] for r in rows)
    by_type, prec = {}, {}
    for layer in layers:
        t = layer.get("LayerType", "?")
        by_type[t] = by_type.get(t, 0) + 1
        p = _precision_of(layer)
        prec[p] = prec.get(p, 0) + 1

    # Time spent in layers that produced fp32 output. In an fp16 engine that is
    # either a deliberate fallback or a missed opportunity, and either way it
    # is worth knowing how much it costs before arguing about it.
    kind = {l.get("Name"): _precision_of(l) for l in layers if l.get("Name")}
    float_share = sum(r["share"] for r in rows if kind.get(r["name"]) == "Float")

    # Time in layers that move or reshape data without computing anything.
    # Reported as one number because the individual layers are usually small
    # and each one alone looks like nothing: moge_2 has 89 Reformats, none
    # above 1.1%, together with its Slices they are 32% of its runtime.
    type_of = {l.get("Name"): l.get("LayerType", "?") for l in layers if l.get("Name")}
    move_share = sum(r["share"] for r in rows
                     if type_of.get(r["name"]) in MOVEMENT_TYPES)

    return {
        "layers_profiled": len(rows),
        "layers_in_engine": len(layers),
        "profile_total_ms": round(prof_total, 4),
        "benchmark_mean_ms": mean_ms,
        "top10_share": round(sum(r["share"] for r in rows[:10]), 4),
        "fp32_output_share": round(float_share, 4),
        "movement_share": round(move_share, 4),
        "layer_types": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "layer_precisions": dict(sorted(prec.items(), key=lambda kv: -kv[1])),
    }


SUSPECT_TYPES = {"Reformat", "Cast", "Transpose", "Shuffle", "Copy", "Resize",
                 "Slice", "Concatenation"}


def suspects(rows, layers, threshold=0.01):
    """Layers that cost real time while doing no arithmetic.

    A Reformat or Shuffle is TensorRT moving data because two neighbours
    disagree about layout or type. One or two are normal. A list of them
    holding several percent of the runtime means the graph is fighting itself,
    and that is the shape the float64 problem took on unidepth_v2.
    """
    kind = {}
    for layer in layers:
        name = layer.get("Name")
        if name:
            kind[name] = layer.get("LayerType", "?")
    found = []
    for r in rows:
        t = kind.get(r["name"], "?")
        if r["share"] >= threshold and (t in SUSPECT_TYPES or t == "?"
                                        and any(s.lower() in r["name"].lower()
                                                for s in SUSPECT_TYPES)):
            found.append(dict(r, layer_type=t))
    return found


def render(model, rows, summary, susp, top=15):
    L = [f"{model}", "=" * len(model), ""]
    L.append(f"layers          {summary['layers_in_engine']} in engine, "
             f"{summary['layers_profiled']} timed")
    if summary.get("benchmark_mean_ms"):
        L.append(f"benchmark       {summary['benchmark_mean_ms']:.3f} ms "
                 f"(profiling serialises layers, so the total below runs high)")
    L.append(f"profile total   {summary['profile_total_ms']:.3f} ms")
    L.append(f"top 10 layers   {summary['top10_share'] * 100:.1f}% of the time")
    L.append("")
    L.append("precision mix   " + ", ".join(f"{k}:{v}" for k, v in
                                            summary["layer_precisions"].items()))
    if summary.get("fp32_output_share"):
        L.append(f"fp32 output     {summary['fp32_output_share'] * 100:.1f}% "
                 f"of the time is in layers producing Float")
    L.append("layer types     " + ", ".join(f"{k}:{v}" for k, v in
                                            list(summary["layer_types"].items())[:8]))
    L.append("")
    L.append(f"{'ms':>9} {'share':>7}  layer")
    L.append(f"{'-' * 9} {'-' * 7}  {'-' * 50}")
    for r in rows[:top]:
        L.append(f"{r['ms']:>9.4f} {r['share'] * 100:>6.2f}%  {r['name'][:70]}")

    if susp:
        L += ["", "data movement holding >=1% each "
                  "(these do no arithmetic; several means the graph is fighting itself)"]
        for r in susp:
            L.append(f"{r['ms']:>9.4f} {r['share'] * 100:>6.2f}%  "
                     f"[{r['layer_type']}] {r['name'][:60]}")
    return "\n".join(L)


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
