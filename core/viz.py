"""Drawing twelve models' output side by side without flattering any of them.

The easy way to draw this comparison is also the dishonest one: colour every
result by its own min and max. Do that and a model that says the wall is 2.4 m
away and a model that says 6.1 m produce the same picture, because the only
thing left on screen is the *shape* of the depth map. The metric difference --
the thing reports/gt.md spends a whole table on -- is normalised away, and the
reader is told the models agree.

So the rule this module exists to enforce is:

* models whose spec.json says ``depth_scale: metric`` are drawn on **one**
  colour range, in metres, with **one** colourbar. Their disagreement is
  visible because it is the same axis;
* models that say ``relative`` are drawn on their own axis, in a different
  colour map, and the panel says on its face that it was normalised or what it
  was fitted to. A relative model claims a shape, not a distance, and the
  picture has to claim the same thing;
* models that say ``unknown`` say so.

Nothing here imports tensorrt, torch, or anything that needs a GPU: the whole
module has to run on a laptop with only the saved arrays, because that is the
only place the layout can actually be checked. The matplotlib backend is Agg,
set at import, so there is no headless-only code path that differs from the
one people look at.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core import preprocess as pre_mod  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Colour maps. Metric and relative deliberately do not share one: two panels
# that cannot be compared should not look like they can.
METRIC_CMAP = "turbo"
RELATIVE_CMAP = "magma"
UNKNOWN_CMAP = "bone"
QUALITATIVE_CMAP = "turbo"
INVALID_RGB = (60, 60, 66)

BANDS = {
    "metric": ("metric depth", METRIC_CMAP, "#3d7dd8"),
    # A relative model the caller asked to be fitted to a metric one. It gets
    # the metric axis so it can be compared and a different frame colour so it
    # is not mistaken for a model that claims metres.
    "fitted": ("relative, fitted", METRIC_CMAP, "#7a4fa3"),
    "relative": ("relative depth", RELATIVE_CMAP, "#c46a1f"),
    "unknown": ("scale unknown", UNKNOWN_CMAP, "#8a8a8a"),
    "qualitative": ("shape comparison", QUALITATIVE_CMAP, "#3f6f8f"),
    "input": ("input", None, "#4d4d4d"),
    "skipped": ("not shown", None, "#a02b2b"),
}

# Which spec output carries a point map, and how it is laid out. Transcribed
# from evaluate_gt.py's adapters -- _point_map_depth reshapes to (h, w, 3) and
# _point_channel_depth to (3, h, w) -- because the flat array is the same size
# either way and guessing would silently transpose the scene.
# tests/test_viz.py checks each name here still has the matching adapter.
POINT_LAYOUT = {
    "moge_2": "hw3",
    "metric_anything": "hw3",
    "unidepth_v2": "chw",
    "unik3d": "chw",
}


# ---------------------------------------------------------------------------
# Metadata: what has to appear next to every panel
# ---------------------------------------------------------------------------

def engine_fingerprint(run, engine_path=None):
    """(short digest, where it came from) for the engine behind a run.

    Preferred is the engine file itself, which is what the desktop can do. A
    laptop has the benchmark record and nothing else, so it falls back to
    digesting the fields the record does carry -- path, size, mtime, ONNX hash
    -- and says ``record`` so nobody reads it as a hash of the bytes. Two
    engines that differ produce different records; the fallback is weaker than
    a file hash but it is not a guess.
    """
    path = engine_path or (run or {}).get("engine_path") or ""
    if path and os.path.isfile(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()[:12], "engine sha256"
    if not run:
        return "unknown", "no record"
    parts = [
        os.path.basename(path),
        str(run.get("engine_bytes") or ""),
        str(run.get("engine_mtime") or ""),
        str(run.get("onnx_sha256") or ""),
        str(run.get("precision") or ""),
        f"{run.get('input_h')}x{run.get('input_w')}",
    ]
    if not any(parts[:4]):
        return "unknown", "no record"
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], "record"


def output_meanings(spec):
    return [(o["name"], o["meaning"]) for o in spec.get("outputs", [])]


def model_card(name, spec, run=None):
    """Everything the panel has to state, read from JSON rather than typed.

    P7's completion condition names six of these: model, input size, what the
    output means, its unit, the precision, and which engine produced it. They
    are gathered in one place so a panel cannot be drawn without them.
    """
    target = (spec.get("build_targets") or [{}])[0]
    profile = (run or {}).get("profile") or target.get("profile")
    if profile and profile in spec.get("profiles", {}):
        h, w = spec["profiles"][profile]["size"]
    else:
        h, w = (None, None)
    if (run or {}).get("input_h") and (run or {}).get("input_w"):
        # The engine that was actually measured wins over the profile: a size
        # sweep leaves records the spec never mentions.
        h, w = run["input_h"], run["input_w"]
    scale = spec.get("depth_scale") or "unknown"
    fp, fp_src = engine_fingerprint(run)
    stats = (run or {}).get("stats") or {}
    return {
        "model": name,
        "input_h": h,
        "input_w": w,
        "band": scale if scale in ("metric", "relative") else "unknown",
        "depth_scale": scale,
        "output_form": spec.get("output_form") or "depth",
        "unit": "m" if scale == "metric" else "-",
        "precision": (run or {}).get("precision") or target.get("precision") or "?",
        "profile": profile or "?",
        "outputs": output_meanings(spec),
        "engine_fingerprint": fp,
        "engine_fingerprint_source": fp_src,
        "engine_path": (run or {}).get("engine_path") or "",
        "engine_present": bool((run or {}).get("engine_path")
                               and os.path.isfile(run["engine_path"])),
        "mean_ms": stats.get("mean_ms"),
        "device": (run or {}).get("device"),
        "preprocess": spec.get("preprocess", {}),
        "caveats": list(spec.get("caveats", [])),
    }


def caption_for(card, extra_lines=()):
    """The small print under a panel. ASCII only; this also lands on a console."""
    size = (f"{card['input_h']}x{card['input_w']}"
            if card["input_h"] else "size unknown")
    unit = card["unit"]
    lines = [
        f"{card['band']} [{unit}]  {size}  {card['precision']}",
        "out: " + ", ".join(f"{n}={m}" for n, m in card["outputs"]),
        f"eng {card['engine_fingerprint']} ({card['engine_fingerprint_source']})",
    ]
    lines.extend(extra_lines)
    return "\n".join(str(x) for x in lines if x)


# How many monospace characters fit under one panel at the sizes render_figure
# uses. Measured by looking at the output, not derived: matplotlib's monospace
# metrics depend on the font it finds, so the honest thing is a margin.
CAPTION_CHARS = 60


def wrap_caption(text, width=CAPTION_CHARS, max_lines=14):
    """Caption text folded to the panel width, continuation lines indented.

    Without this the third model's caption runs under the fourth model's
    picture, and the reader attributes an engine fingerprint to the wrong
    model -- which is worse than not printing it.
    """
    out = []
    for line in str(text).splitlines():
        if len(line) <= width:
            out.append(line)
            continue
        words, cur = [], ""
        for word in line.split(" "):
            # A path or a hash is one word and longer than the panel. Broken
            # rather than allowed to run into the next model's picture.
            while len(word) > width - 2:
                words.append(word[:width - 2])
                word = word[width - 2:]
            words.append(word)
        for word in words:
            candidate = (cur + " " + word) if cur else word
            if len(candidate) > width and cur:
                out.append(cur)
                cur = "  " + word
            else:
                cur = candidate
        if cur:
            out.append(cur)
    if len(out) > max_lines:
        out = out[:max_lines - 1] + ["  ..."]
    return out


# ---------------------------------------------------------------------------
# Geometry: how the source image was mapped into the network input
# ---------------------------------------------------------------------------

def input_geometry(spec, src_h, src_w, dst_h, dst_w):
    """A preprocess.Geometry for the resize type spec.json declares.

    Only two types are declared across the repository: ``stretch``, which fills
    the input exactly and pads nothing, and ``keep_ratio_pad``, which scales by
    the smaller ratio and centres the result. The second is the one with a
    pad region to draw, and it has to agree with evaluate_gt's
    _metric3d_geometry down to the odd pixel -- tests/test_viz.py checks that
    against the adapter rather than trusting this comment.
    """
    kind = (spec.get("preprocess") or {}).get("type", "stretch")
    if kind == "keep_ratio_pad":
        scale = min(dst_h / src_h, dst_w / src_w)
        inner_h, inner_w = int(src_h * scale), int(src_w * scale)
        pad_h, pad_w = dst_h - inner_h, dst_w - inner_w
        return pre_mod.Geometry(
            src_h=src_h, src_w=src_w, dst_h=dst_h, dst_w=dst_w,
            inner_h=inner_h, inner_w=inner_w,
            pad_top=pad_h // 2, pad_bottom=pad_h - pad_h // 2,
            pad_left=pad_w // 2, pad_right=pad_w - pad_w // 2,
        )
    if kind != "stretch":
        raise ValueError(f"no geometry for declared preprocess type {kind!r}")
    return pre_mod.Geometry(src_h=src_h, src_w=src_w, dst_h=dst_h, dst_w=dst_w,
                            inner_h=dst_h, inner_w=dst_w)


def geometry_note(geom):
    """One ASCII line describing what preprocessing did to the frame.

    Every size here is height x width, the order spec.json, the benchmark
    records and the engine filenames all use. Mixing WxH into a figure whose
    captions say HxW is how 388x518 gets read as a portrait model.
    """
    pads = (geom.pad_top, geom.pad_bottom, geom.pad_left, geom.pad_right)
    if not any(pads):
        return (f"hxw: src {geom.src_h}x{geom.src_w} -> {geom.dst_h}x{geom.dst_w} "
                f"stretched, no pad")
    return (f"hxw: src {geom.src_h}x{geom.src_w} -> inner "
            f"{geom.inner_h}x{geom.inner_w} + pad t{pads[0]} b{pads[1]} "
            f"l{pads[2]} r{pads[3]}")


def preview_input_tensor(blob):
    """A network input tensor as something a person can look at.

    Per-tensor min/max is fine here and only here: this is the image that went
    in, not a measurement coming out, and the normalisation each model applies
    is its own. The pad region stays visible because it is a constant, which is
    the whole reason to draw it.
    """
    a = np.asarray(blob, dtype=np.float64)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"expected a CHW or HWC tensor, got shape {np.shape(blob)}")
    if a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = a.transpose(1, 2, 0)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros(a.shape[:2] + (3,), np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Colouring
# ---------------------------------------------------------------------------

def robust_range(arrays, lo_pct=2.0, hi_pct=98.0):
    """One (vmin, vmax) over every array given, ignoring nan and non-positive.

    Pooled, not per-array: this is the shared metric axis, and computing it
    per model would be the very normalisation this module exists to refuse.
    """
    pool = []
    for a in arrays:
        v = np.asarray(a, dtype=np.float64).ravel()
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            # Subsample: the pooled percentile over eleven 3-megapixel maps is
            # slow and the answer does not change.
            pool.append(v if v.size <= 200000 else v[:: max(1, v.size // 200000)])
    if not pool:
        return 0.0, 1.0
    allv = np.concatenate(pool)
    vmin, vmax = np.percentile(allv, [lo_pct, hi_pct])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(allv.min()), float(allv.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)


def colorize(arr, vmin, vmax, cmap=METRIC_CMAP, invalid_rgb=INVALID_RGB):
    """A depth map as RGB uint8 on a *given* range. Never on its own range."""
    a = np.asarray(arr, dtype=np.float64)
    bad = ~np.isfinite(a)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    t = np.clip((np.nan_to_num(a, nan=vmin) - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = (matplotlib.colormaps[cmap](t)[..., :3] * 255.0).astype(np.uint8)
    rgb[bad] = np.asarray(invalid_rgb, np.uint8)
    return rgb


def fit_scale_shift(pred, ref, mask=None):
    """Least squares (s, t) with s*pred + t ~= ref, over finite pixels.

    Used only when the caller asks for a relative model to be put on another
    model's axis, and the panel then says so. There is no ground truth in a
    demo, so this borrows the reference's errors along with its scale; that is
    a statement the picture has to make, not hide.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {r.shape}")
    m = np.isfinite(p) & np.isfinite(r)
    if mask is not None:
        m &= np.asarray(mask).ravel().astype(bool)
    if m.sum() < 16:
        raise ValueError("fewer than 16 pixels to fit a scale and a shift on")
    p, r = p[m], r[m]
    A = np.stack([p, np.ones_like(p)], axis=1)
    sol, *_ = np.linalg.lstsq(A, r, rcond=None)
    return float(sol[0]), float(sol[1])


def display_ranges(cards, depths, override=None):
    """({model: (vmin, vmax, cmap, panel note)}, the shared range's one line).

    The contract this function is: **every metric model gets the identical
    (vmin, vmax, cmap)**. tests/test_viz.py asserts it, because it is the one
    property that stops the figure from lying.
    """
    metric = [m for m in depths if cards[m]["band"] == "metric"]
    if override is not None:
        shared = (float(override[0]), float(override[1]))
        shared_note = (f"metric range fixed by --depth-range "
                       f"{shared[0]:g}-{shared[1]:g} m")
    else:
        shared = robust_range([depths[m] for m in metric])
        shared_note = (f"shared metric range {shared[0]:.2f}-{shared[1]:.2f} m "
                       f"(2-98 pct pooled over {len(metric)} metric models)")
    out = {}
    for m, d in depths.items():
        band = cards[m]["band"]
        if band in ("metric", "fitted"):
            # `fitted` shares the axis but never sets it: the range is pooled
            # over models that actually claim metres, so a borrowed scale
            # cannot move the axis everyone else is read against.
            out[m] = (shared[0], shared[1], METRIC_CMAP,
                      f"shared metric axis {shared[0]:.2f}-{shared[1]:.2f} m "
                      f"(see the bar)")
        else:
            lo, hi = robust_range([d])
            cmap = RELATIVE_CMAP if band == "relative" else UNKNOWN_CMAP
            out[m] = (lo, hi, cmap, f"own range {lo:.3g}-{hi:.3g}, not metres")
    return out, shared_note


def qualitative_display(depth, output_form="depth"):
    """Return a near-is-high map and its own robust display range.

    This is intentionally a *shape-only* representation for a compact README
    gallery. It must not be used to compare metric values: each model receives
    its own 2--98 percentile range. Direct depth is converted to proximity;
    models that already output inverse depth keep their orientation. As a
    result, the same Turbo colours mean the same visual direction in every
    panel: warm is near and cool is far.
    """
    d = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(d) & (d > 0)
    shown = np.full(d.shape, np.nan, dtype=np.float64)
    if output_form == "inverse_depth":
        shown[valid] = d[valid]
    else:
        shown[valid] = 1.0 / d[valid]
    lo, hi = robust_range([shown])
    return shown, lo, hi


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass
class Panel:
    title: str
    image: np.ndarray                      # HxWx3 uint8
    caption: str = ""
    band: str = "metric"
    badge: str = ""                        # e.g. SYNTHETIC
    box: Optional[Tuple[int, int, int, int]] = None   # (y0, y1, x0, x1)
    box_label: str = ""


@dataclass
class Section:
    header: str
    panels: Sequence[Panel] = field(default_factory=list)
    colorbar: Optional[dict] = None        # {"vmin","vmax","cmap","label"}
    note: str = ""


def _draw_panel(fig, cell, panel, cap_lines):
    """One panel into one gridspec cell: picture on top, caption below it.

    The caption gets its own sub-cell rather than being written under the axes
    in axes coordinates. With one axes per cell, a portrait image keeps the
    full cell height (imshow preserves the pixel aspect by shrinking width),
    the caption lands in the row underneath, and every label in the figure ends
    up attached to the wrong picture. That is what this split prevents.
    """
    sub = cell.subgridspec(2, 1, height_ratios=[1.0, max(0.14, LINE_IN * cap_lines
                                                         + 0.10)], hspace=0.04)
    ax = fig.add_subplot(sub[0])
    ax.set_anchor("N")
    ax.imshow(panel.image, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    colour = BANDS.get(panel.band, BANDS["input"])[2]
    for s in ax.spines.values():
        s.set_edgecolor(colour)
        s.set_linewidth(1.6)
    ax.set_title(panel.title, fontsize=8.5, color=colour, pad=3, loc="left")
    if panel.box:
        y0, y1, x0, x1 = panel.box
        ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                   fill=False, edgecolor="#ffffff",
                                   linewidth=1.0, linestyle="--"))
        if panel.box_label:
            ax.text(x0 + 3, y0 + 3, panel.box_label, fontsize=5.5,
                    color="#ffffff", va="top", ha="left",
                    bbox=dict(facecolor="#000000", alpha=0.45, pad=1, lw=0))
    if panel.badge:
        ax.text(0.98, 0.02, panel.badge, transform=ax.transAxes, fontsize=7,
                color="#ffffff", ha="right", va="bottom", family="monospace",
                bbox=dict(facecolor="#a02b2b", alpha=0.85, pad=1.5, lw=0))
    cap = fig.add_subplot(sub[1])
    cap.axis("off")
    if panel.caption:
        cap.text(0.0, 1.0, "\n".join(wrap_caption(panel.caption)),
                 transform=cap.transAxes, fontsize=5.6, family="monospace",
                 va="top", ha="left", color="#222222")
    return ax


LINE_IN = 0.088          # vertical space one caption line needs, in inches


def render_figure(sections, title, subtitle="", footer="", cols=4,
                  panel_in=2.9):
    """Sections of panels into one matplotlib Figure. No pyplot state relied on.

    Rows are laid out by hand rather than with subplots because the sections
    have headers and one of them owns a colourbar, and a figure whose metric
    colourbar drifted onto the relative section would be worse than no
    colourbar at all.

    Row height follows the longest caption in that row rather than a constant,
    because the captions are the part carrying the model name, the input size
    and the engine fingerprint, and a caption that runs into the next row is
    read as belonging to the wrong model.
    """
    heights, plan = [], []
    for sec in sections:
        if not sec.panels:
            continue
        heights.append(0.34)
        plan.append(("header", sec))
        nrows = (len(sec.panels) + cols - 1) // cols
        for r in range(nrows):
            row = sec.panels[r * cols:(r + 1) * cols]
            lines = max((len(wrap_caption(p.caption)) for p in row), default=0)
            heights.append(panel_in + LINE_IN * lines + 0.20)
            plan.append(("panels", (sec, r)))
        if sec.colorbar:
            heights.append(0.62)
            plan.append(("cbar", sec))
    if not plan:
        heights, plan = [1.0], [("empty", None)]

    top_in = 0.95 + (0.24 if subtitle else 0.0)
    bottom_in = 0.30 + 0.16 * (footer.count("\n") + 1 if footer else 0)
    fig_h = sum(heights) + top_in + bottom_in
    fig_w = cols * panel_in + 0.7
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=110)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(len(heights), cols, height_ratios=heights,
                          left=0.03, right=0.985,
                          top=1 - top_in / fig_h, bottom=bottom_in / fig_h,
                          wspace=0.06, hspace=0.10)

    fig.text(0.03, 1 - 0.34 / fig_h, title, fontsize=13, va="top", ha="left",
             fontweight="bold")
    if subtitle:
        fig.text(0.03, 1 - 0.66 / fig_h, subtitle, fontsize=8, va="top",
                 ha="left", color="#333333", family="monospace")

    for i, (kind, payload) in enumerate(plan):
        if kind == "header":
            ax = fig.add_subplot(gs[i, :])
            ax.axis("off")
            colour = BANDS.get(payload.panels[0].band, BANDS["input"])[2]
            ax.text(0, 0.15, payload.header, fontsize=9.5, fontweight="bold",
                    color=colour, va="bottom", ha="left")
            if payload.note:
                ax.text(1.0, 0.15, payload.note, fontsize=7.5, color="#444444",
                        va="bottom", ha="right", family="monospace")
        elif kind == "panels":
            sec, r = payload
            row = sec.panels[r * cols:(r + 1) * cols]
            lines = max((len(wrap_caption(p.caption)) for p in row), default=1)
            for c, panel in enumerate(row):
                _draw_panel(fig, gs[i, c], panel, lines)
        elif kind == "cbar":
            host = fig.add_subplot(gs[i, :])
            host.axis("off")
            # The bar sits in the upper half of its row so its tick labels and
            # its label have somewhere to go that is not the next section's
            # heading. They collided when it filled the row.
            ax = host.inset_axes([0.01, 0.62, 0.98, 0.30])
            cb = payload.colorbar
            norm = matplotlib.colors.Normalize(vmin=cb["vmin"], vmax=cb["vmax"])
            sm = matplotlib.cm.ScalarMappable(norm=norm,
                                              cmap=matplotlib.colormaps[cb["cmap"]])
            bar = fig.colorbar(sm, cax=ax, orientation="horizontal")
            bar.set_label(cb["label"], fontsize=8)
            bar.ax.tick_params(labelsize=7)
        else:
            fig.add_subplot(gs[0, :]).axis("off")

    if footer:
        fig.text(0.03, bottom_in / fig_h * 0.75, footer, fontsize=6.8,
                 va="top", ha="left", color="#444444", family="monospace")
    return fig


def save_figure(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Point clouds
# ---------------------------------------------------------------------------

def points_from_point_map(arr, layout, h, w):
    """A model's point map as (h, w, 3) metres, whichever way it was stored."""
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    if a.size != h * w * 3:
        raise ValueError(f"point map has {a.size} values, expected {h * w * 3}")
    if layout == "hw3":
        return a.reshape(h, w, 3)
    if layout == "chw":
        return a.reshape(3, h, w).transpose(1, 2, 0)
    raise ValueError(f"unknown point map layout {layout!r}")


def points_from_depth(depth, fx, fy=None, cx=None, cy=None):
    """Unproject a depth map with a pinhole camera. Right-handed, +z forward."""
    d = np.asarray(depth, dtype=np.float64)
    h, w = d.shape[:2]
    fy = fx if fy is None else fy
    cx = (w - 1) / 2.0 if cx is None else cx
    cy = (h - 1) / 2.0 if cy is None else cy
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    return np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], axis=-1)


def focal_from_fov(fov_deg, width):
    """Pixels of focal length from a horizontal field of view."""
    return 0.5 * float(width) / np.tan(0.5 * np.deg2rad(float(fov_deg)))


def subsample_cloud(xyz, rgb=None, max_points=120000, seed=0):
    """Flatten to (N, 3) valid points, thinned deterministically."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    col = (np.asarray(rgb).reshape(-1, 3) if rgb is not None
           else np.full((pts.shape[0], 3), 200, np.uint8))
    keep = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
    pts, col = pts[keep], col[keep]
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(seed).choice(pts.shape[0], max_points,
                                                 replace=False)
        idx.sort()
        pts, col = pts[idx], col[idx]
    return pts, col.astype(np.uint8)


def render_points(xyz, rgb=None, size=(320, 420), elev_deg=-18.0, azim_deg=24.0,
                  point_px=1, bg=(22, 22, 26)):
    """An orthographic splat of a point cloud, with a z-buffer.

    matplotlib's 3D scatter would do this, but it has no depth buffer -- points
    are drawn in list order, so the back of the room paints over the front and
    the picture is wrong in a way that looks fine. Fifteen lines of numpy get it
    right and run headless.
    """
    h, w = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = np.asarray(bg, np.uint8)
    pts, col = subsample_cloud(xyz, rgb)
    if pts.shape[0] == 0:
        return img
    centre = np.median(pts, axis=0)
    p = pts - centre
    a, e = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    p = p @ ry.T @ rx.T
    span = np.percentile(np.abs(p[:, :2]), 99.0)
    if not np.isfinite(span) or span <= 0:
        # Everything projects to one column -- a cloud seen exactly end-on, or
        # a handful of collinear points. Still drawn, because a blank tile
        # would read as "no cloud" rather than "a degenerate view of one".
        span = 1.0
    scale = 0.46 * min(h, w) / span
    u = np.round(p[:, 0] * scale + w / 2.0).astype(np.int64)
    v = np.round(p[:, 1] * scale + h / 2.0).astype(np.int64)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z, col = u[ok], v[ok], p[ok, 2], col[ok]
    order = np.argsort(-z)          # far first, so near overwrites it
    u, v, col = u[order], v[order], col[order]
    for dy in range(-point_px + 1, point_px):
        for dx in range(-point_px + 1, point_px):
            yy, xx = np.clip(v + dy, 0, h - 1), np.clip(u + dx, 0, w - 1)
            img[yy, xx] = col
    return img


def write_ply(path, xyz, rgb=None):
    """Binary little-endian PLY. tools/retired/vis_ply.py already reads this repo's clouds."""
    pts, col = subsample_cloud(xyz, rgb, max_points=10 ** 9)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rec = np.zeros(pts.shape[0], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"),
                                        ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["red"], rec["green"], rec["blue"] = col[:, 0], col[:, 1], col[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())
    return path


# ---------------------------------------------------------------------------
# Auxiliary output views
# ---------------------------------------------------------------------------

def view_of_output(name, meaning, arr, eh, ew, oh, ow):
    """An RGB view of a non-depth output, or None when it is not an image.

    `meaning` comes from spec.json, so a model that does not declare a mask
    does not get a mask panel invented for it.
    """
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    text = f"{meaning}"
    if a.size == eh * ew:
        m = a.reshape(eh, ew)
        if "mask" in meaning:
            img = np.where(m > 0.5, 235, 40).astype(np.uint8)
            rgb = np.repeat(img[..., None], 3, axis=-1)
            text = f"{meaning}: {float((m > 0.5).mean()) * 100:.1f}% valid"
        else:
            lo, hi = robust_range([m])
            rgb = colorize(m, lo, hi, "viridis")
            text = f"{meaning}: {lo:.3g}..{hi:.3g}"
        return cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_NEAREST), text
    if a.size == eh * ew * 3:
        v = a.reshape(eh, ew, 3) if a.size // 3 == eh * ew else None
        if v is None:
            return None, text
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if hi - lo < 1e-12:
            hi = lo + 1.0
        rgb = ((v - lo) / (hi - lo) * 255).astype(np.uint8)
        return cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_NEAREST), \
            f"{meaning}: {lo:.3g}..{hi:.3g}"
    if a.size <= 16:
        vals = ", ".join(f"{x:.4g}" for x in a[:9])
        return None, f"{meaning}: {vals}"
    return None, f"{meaning}: {a.size} values, not an image"


# ---------------------------------------------------------------------------
# Saved outputs
# ---------------------------------------------------------------------------

def save_outputs(path, model, spec, outputs, meta=None):
    """One model's raw engine outputs as an npz keyed by the spec's names."""
    names = [o["name"] for o in spec.get("outputs", [])]
    data = {}
    for i, arr in enumerate(outputs):
        key = names[i] if i < len(names) else f"out{i}"
        data[key] = np.asarray(arr)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **data)
    side = os.path.splitext(path)[0] + ".json"
    with open(side, "w", encoding="utf-8") as f:
        json.dump(dict(meta or {}, model=model, keys=list(data),
                       written=datetime.now(timezone.utc).isoformat(timespec="seconds")),
                  f, indent=2)
    return path


def load_outputs(path, spec):
    """An npz or npy back into the positional list the adapters expect."""
    if path.endswith(".npy"):
        return [np.load(path)]
    with np.load(path) as z:
        keys = list(z.files)
        names = [o["name"] for o in spec.get("outputs", [])]
        if all(n in keys for n in names) and names:
            return [z[n] for n in names]
        numbered = [k for k in keys if k.startswith("out") and k[3:].isdigit()]
        if numbered:
            return [z[k] for k in sorted(numbered, key=lambda s: int(s[3:]))]
        return [z[k] for k in keys]


def find_saved(root, image_key, model):
    """Where a saved output for this (image, model) lives, or None.

    Three spellings are accepted because the array can arrive from a desktop
    run, from a one-image scratch directory, or from this file's own writer.
    """
    for rel in (os.path.join(image_key, model + ".npz"),
                f"{model}__{image_key}.npz",
                model + ".npz",
                os.path.join(image_key, model + ".npy"),
                model + ".npy"):
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return None
