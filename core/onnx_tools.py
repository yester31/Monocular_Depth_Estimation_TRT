"""Post-export fixes applied to the ONNX file itself.

Kept separate from core/export_compat.py: those patch the model while it is
being traced, these rewrite the graph afterwards. Reach for a graph rewrite
only when the source-level fix would mean touching many places in an upstream
clone -- editing the model is easier to read and easier to verify.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

DOUBLE = TensorProto.DOUBLE
FLOAT = TensorProto.FLOAT


def demote_float64(path, out_path=None, verbose=True):
    """Rewrite every float64 tensor in the graph as float32.

    Several exports here contain float64 because upstream deliberately uses it
    for precision -- VGGT's positional-embedding frequencies, UniDepth's
    coordinate grids -- and casts back afterwards. PyTorch handles the mixed
    arithmetic by type promotion. ONNX does not: an operator's type parameter
    binds to a single type, so the exported graph is invalid and onnxruntime
    refuses to load it outright:

        Type Error: Type parameter (T) of Optype (Einsum) bound to different
        types (tensor(float) and tensor(double))

    **This does not change what the engine computes.** TensorRT has no fp64,
    so it was already doing this silently; the rewrite only makes the file say
    what the engine does, which is what lets other tools -- including the
    accuracy check in verify_accuracy.py -- read it at all.

    It does change the graph relative to PyTorch, in exactly the places
    upstream asked for extra precision. Whether that matters is measurable:
    compare the engine against the rewritten graph, which is what
    verify_accuracy.py does.

    Returns a count of what was changed.
    """
    model = onnx.load(path)
    g = model.graph
    changed = {"initializer": 0, "constant": 0, "cast": 0, "value_info": 0}

    for init in g.initializer:
        if init.data_type == DOUBLE:
            arr = numpy_helper.to_array(init).astype("float32")
            new = numpy_helper.from_array(arr, init.name)
            init.CopyFrom(new)
            changed["initializer"] += 1

    for node in g.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == DOUBLE:
                    arr = numpy_helper.to_array(a.t).astype("float32")
                    a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))
                    changed["constant"] += 1
        elif node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == DOUBLE:
                    a.i = FLOAT
                    changed["cast"] += 1

    for coll in (g.input, g.output, g.value_info):
        for v in coll:
            if v.type.tensor_type.elem_type == DOUBLE:
                v.type.tensor_type.elem_type = FLOAT
                changed["value_info"] += 1

    total = sum(changed.values())
    if verbose:
        if total:
            print(f"[MDET] float64 -> float32: " +
                  ", ".join(f"{k} {v}" for k, v in changed.items() if v))
        else:
            print("[MDET] no float64 in the graph")

    if total:
        onnx.save(model, out_path or path)
    return changed


def add_uint8_input(path, out_path, mean, std, *, scale=255.0, input_name=None,
                    new_name="image_u8", verbose=True):
    """Prepend cast + normalize + transpose so the graph takes uint8 NHWC.

    The engine currently takes float32 NCHW and the caller does the arithmetic:
    divide by 255, subtract the mean, divide by the standard deviation,
    transpose. Moving that inside cuts the host-to-device copy to a quarter --
    3.1 MB to 0.8 MB at 518 -- but adds work to the graph, and there is no
    guarantee TensorRT fuses it into the first convolution. Which effect wins
    is a measurement, not a prediction, which is what ab_input_dtype.py does.

    Rewriting the exported graph rather than changing each onnx_export.py is
    deliberate: it keeps the body of the two graphs byte-identical, so a
    difference in the measurement can only come from the preamble. Exporting
    twice would also re-run the tracer, and three of these models needed
    export-time patches to trace at all.

    The layout is NHWC because that is what an image already is in memory --
    NCHW uint8 would need the same transpose on the host and save nothing.

    `mean` and `std` are in the units after dividing by `scale`, i.e. the same
    numbers the preprocessing code uses. Pass scale=1.0 for a model that does
    not divide.
    """
    model = onnx.load(path)
    g = model.graph

    src = None
    for v in g.input:
        if input_name is None or v.name == input_name:
            src = v
            break
    if src is None:
        raise ValueError(f"{path}: no input named {input_name!r}")

    shape = [d.dim_value or d.dim_param or 0
             for d in src.type.tensor_type.shape.dim]
    rank = len(shape)
    # Rank 4 is [N,3,H,W]; rank 5 is VGGT's [N,S,3,H,W], where the sequence
    # axis sits in front and the rest is the same picture. Both become
    # channels-last by moving the 3 to the end, which is the only difference
    # between them.
    if rank not in (4, 5):
        raise ValueError(f"{path}: expected a rank-4 or rank-5 input, got {shape}")
    ch = rank - 3
    if shape[ch] != 3:
        raise ValueError(f"{path}: expected 3 channels at axis {ch}, got {shape}")
    lead, h, w = shape[:ch], shape[ch + 1], shape[ch + 2]
    nhwc_shape = list(lead) + [h, w, 3]
    perm = list(range(ch)) + [rank - 1, ch, ch + 1]

    mean = [float(x) for x in mean]
    std = [float(x) for x in std]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must have three entries")

    # Shaped [1,...,1,3] so they broadcast over channels-last without a reshape.
    def const(name, values):
        return numpy_helper.from_array(
            np.array(values, dtype="float32").reshape([1] * (rank - 1) + [3]), name)

    g.initializer.extend([
        numpy_helper.from_array(np.array(scale, dtype="float32"), "u8_scale"),
        const("u8_mean", mean),
        const("u8_std", std),
    ])

    u8 = onnx.helper.make_tensor_value_info(
        new_name, TensorProto.UINT8, nhwc_shape)

    pre = [
        onnx.helper.make_node("Cast", [new_name], ["u8_f32"], to=FLOAT,
                              name="u8_cast"),
        onnx.helper.make_node("Div", ["u8_f32", "u8_scale"], ["u8_scaled"],
                              name="u8_div"),
        onnx.helper.make_node("Sub", ["u8_scaled", "u8_mean"], ["u8_centred"],
                              name="u8_sub"),
        onnx.helper.make_node("Div", ["u8_centred", "u8_std"], ["u8_norm"],
                              name="u8_norm"),
        onnx.helper.make_node("Transpose", ["u8_norm"], [src.name],
                              perm=perm, name="u8_channels_last_to_first"),
    ]

    # The old input becomes an ordinary intermediate value: same name, so
    # nothing downstream changes, but no longer declared as a graph input.
    g.input.remove(src)
    g.input.insert(0, u8)
    # protobuf repeated fields have no slice assignment, so the body is copied
    # out, the field cleared, and everything put back in order.
    body = list(g.node)
    del g.node[:]
    g.node.extend(pre + body)

    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    if verbose:
        print(f"[MDET] uint8 input {new_name} {nhwc_shape} -> {src.name} "
              f"{shape}  ({os.path.basename(out_path)})")
    return out_path


def count_float64(path):
    """How much float64 a graph still holds. For checking a fix landed."""
    g = onnx.load(path, load_external_data=False).graph
    n = sum(1 for i in g.initializer if i.data_type == DOUBLE)
    for node in g.node:
        if node.op_type == "Constant":
            n += sum(1 for a in node.attribute
                     if a.name == "value" and a.t.data_type == DOUBLE)
        elif node.op_type == "Cast":
            n += sum(1 for a in node.attribute if a.name == "to" and a.i == DOUBLE)
    for coll in (g.input, g.output, g.value_info):
        n += sum(1 for v in coll if v.type.tensor_type.elem_type == DOUBLE)
    return n
