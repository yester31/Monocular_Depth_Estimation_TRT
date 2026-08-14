"""The uint8 preamble must compute what the host preprocessing computed.

Moving normalization into the graph is only interesting if it is the same
arithmetic. If it is not, a faster engine is just a different engine, and the
A/B is measuring two things at once.

So this builds a small graph, rewrites it, and runs both under onnxruntime on
the same picture -- one through the host preprocessing, one through the
preamble. Order matters: divide by 255 first, then subtract the mean, then
divide by the standard deviation, because the constants are in [0,1] units.
Getting that order wrong produces output that still looks like a depth map.

Skips where onnx is not installed; the laptop has no model environment.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        _failures.append(name)


try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
    from core.onnx_tools import add_uint8_input
    HAVE = True
except ImportError as e:                                      # noqa: BLE001
    HAVE = False
    WHY = str(e)

# The standalone runner at the bottom checks HAVE before doing anything, but
# pytest calls each test function directly, and without onnx the names imported
# above do not exist -- so six tests failed with NameError on any machine
# lacking it, which reads as a broken test rather than an absent dependency.
try:
    import pytest
    pytestmark = pytest.mark.skipif(
        not HAVE, reason=f"needs onnx and onnxruntime ({WHY if not HAVE else ''})")
except ImportError:                                           # no pytest either
    pass

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
H = W = 8


def _tiny_model(path):
    """A graph whose output depends on every input element, so a wrong
    transpose or a broadcast over the wrong axis cannot cancel out."""
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, H, W])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, H, W])
    w = onnx.numpy_helper.from_array(
        (np.arange(3 * H * W, dtype="float32").reshape(1, 3, H, W) * 0.01), "w")
    node = helper.make_node("Mul", ["input", "w"], ["output"])
    g = helper.make_graph([node], "tiny", [x], [y], initializer=[w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return path


def _run(path, name, arr):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s.run(None, {name: arr})[0]


def test_preamble_matches_host_preprocessing():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = _tiny_model(os.path.join(td, "tiny.onnx"))
        u8_path = os.path.join(td, "tiny_u8.onnx")
        add_uint8_input(base, u8_path, MEAN, STD, verbose=False)

        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(1, H, W, 3), dtype=np.uint8)

        # what the scripts do on the host today
        f = img.astype(np.float32) / 255.0
        f = (f - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        nchw = np.ascontiguousarray(f.transpose(0, 3, 1, 2))

        ref = _run(base, "input", nchw)
        got = _run(u8_path, "image_u8", img)

        check("same shape", ref.shape == got.shape, f"{ref.shape} vs {got.shape}")
        diff = float(np.abs(ref - got).max())
        scale = float(np.abs(ref).max()) or 1.0
        check("same values", diff / scale < 1e-6, f"max rel diff {diff / scale:.2e}")


def test_the_graph_body_is_untouched():
    """The point of rewriting rather than re-exporting: only the preamble
    differs, so a timing difference cannot come from anywhere else."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = _tiny_model(os.path.join(td, "tiny.onnx"))
        u8_path = os.path.join(td, "tiny_u8.onnx")
        add_uint8_input(base, u8_path, MEAN, STD, verbose=False)

        before = [n.op_type for n in onnx.load(base).graph.node]
        after = [n.op_type for n in onnx.load(u8_path).graph.node]
        check("body unchanged", after[-len(before):] == before, str(after))
        check("preamble is five nodes", len(after) - len(before) == 5, str(after))


def test_identity_steps_are_left_out():
    """VGGT divides by 255 and nothing else. A mean of zeros and a std of ones
    would add two operators computing the identity -- TensorRT would probably
    fold them, but 'probably' is what the A/B is measuring."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = _tiny_model(os.path.join(td, "tiny.onnx"))
        out = os.path.join(td, "tiny_u8.onnx")
        add_uint8_input(base, out, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], verbose=False)

        added = [n.op_type for n in onnx.load(out).graph.node][:-1]
        added = [op for op in added if op in ("Cast", "Div", "Sub", "Transpose")]
        check("cast, divide by 255, transpose only",
              added[:2] == ["Cast", "Div"], str(added))
        ops = [n.op_type for n in onnx.load(out).graph.node]
        check("no Sub", "Sub" not in ops, str(ops))
        check("exactly three added", len(ops) - 1 == 3, str(ops))

        rng = np.random.default_rng(2)
        img = rng.integers(0, 256, size=(1, H, W, 3), dtype=np.uint8)
        nchw = np.ascontiguousarray(
            (img.astype(np.float32) / 255.0).transpose(0, 3, 1, 2))
        ref, got = _run(base, "input", nchw), _run(out, "image_u8", img)
        check("still the same values",
              float(np.abs(ref - got).max()) / (float(np.abs(ref).max()) or 1) < 1e-6)


def test_input_is_uint8_nhwc():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = _tiny_model(os.path.join(td, "tiny.onnx"))
        u8_path = os.path.join(td, "tiny_u8.onnx")
        add_uint8_input(base, u8_path, MEAN, STD, verbose=False)
        g = onnx.load(u8_path).graph
        check("one input", len(g.input) == 1, str([v.name for v in g.input]))
        v = g.input[0]
        check("uint8", v.type.tensor_type.elem_type == TensorProto.UINT8)
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        check("NHWC", dims == [1, H, W, 3], str(dims))


def test_a_sequence_input_gets_the_same_preamble():
    """VGGT's input is [1,S,3,H,W]. The only difference from a plain image is
    where the channel axis sits, so it gets the same five nodes with a longer
    permutation rather than a special case."""
    import tempfile
    S = 2
    with tempfile.TemporaryDirectory() as td:
        x = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                          [1, S, 3, H, W])
        y = helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                          [1, S, 3, H, W])
        w = onnx.numpy_helper.from_array(
            (np.arange(S * 3 * H * W, dtype="float32").reshape(1, S, 3, H, W)
             * 0.01), "w")
        m = helper.make_model(
            helper.make_graph([helper.make_node("Mul", ["input", "w"], ["output"])],
                              "seq", [x], [y], initializer=[w]),
            opset_imports=[helper.make_opsetid("", 17)])
        base = os.path.join(td, "seq.onnx")
        onnx.save(m, base)
        out = os.path.join(td, "seq_u8.onnx")
        add_uint8_input(base, out, MEAN, STD, verbose=False)

        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, size=(1, S, H, W, 3), dtype=np.uint8)
        f = img.astype(np.float32) / 255.0
        f = (f - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        nschw = np.ascontiguousarray(f.transpose(0, 1, 4, 2, 3))

        ref = _run(base, "input", nschw)
        got = _run(out, "image_u8", img)
        diff = float(np.abs(ref - got).max())
        scale = float(np.abs(ref).max()) or 1.0
        check("same values", diff / scale < 1e-6, f"max rel diff {diff / scale:.2e}")
        dims = [d.dim_value for d in onnx.load(out).graph.input[0]
                .type.tensor_type.shape.dim]
        check("channels last", dims == [1, S, H, W, 3], str(dims))


def test_an_unexpected_rank_is_refused():
    """Anything that is not an image or a sequence of images has no obvious
    channel axis, and guessing one produces a graph that builds and returns
    nonsense."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, H])
        y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, H])
        m = helper.make_model(
            helper.make_graph([helper.make_node("Identity", ["input"], ["output"])],
                              "flat", [x], [y]),
            opset_imports=[helper.make_opsetid("", 17)])
        p = os.path.join(td, "flat.onnx")
        onnx.save(m, p)
        try:
            add_uint8_input(p, os.path.join(td, "out.onnx"), MEAN, STD,
                            verbose=False)
            check("refused", False, "accepted a rank-3 input")
        except ValueError as e:
            check("refused", "rank-4" in str(e), str(e))


if __name__ == "__main__":
    if not HAVE:
        print(f"SKIP  onnx/onnxruntime not installed here ({WHY})")
        sys.exit(0)
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 60)
    print("ALL PASS" if not _failures else f"{len(_failures)} FAILED: {_failures}")
    sys.exit(1 if _failures else 0)
