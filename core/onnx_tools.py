"""Post-export fixes applied to the ONNX file itself.

Kept separate from core/export_compat.py: those patch the model while it is
being traced, these rewrite the graph afterwards. Reach for a graph rewrite
only when the source-level fix would mean touching many places in an upstream
clone -- editing the model is easier to read and easier to verify.
"""

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
