# by yhpark 2026-8-14
# HyDen (MetaDepth) ONNX export
"""Export HyDen-DA2-Large to ONNX.

The first hybrid encoder in this comparison. Every other ViT row is one path
of attention; HyDen runs a CNN path and a ViT path together, so it is the
first graph here that makes TensorRT fuse convolutions and attention in the
same engine. That is the reason it is in the table -- upstream's "up to 10x
faster 4K inference" claim names no GPU and publishes no numbers, and this
repository measures at 518x518 on a pinned clock rather than repeating it.

Needs the upstream clone beside this file:

    cd models/hyden
    git clone https://github.com/facebookresearch/metadepth.git
    mkdir -p metadepth/checkpoints
    # HyDen-DA2-Large from facebook/hyden-da2-relative-depth ->
    #   metadepth/checkpoints/hyden_da2_vitl.pth

Licence: FAIR Noncommercial Research License, and it covers the weights as
well as the code. It permits derivative works and redistribution as long as
the agreement travels with them, so it restricts *use* rather than propagating
to this repository's MIT licence.

**The resize is not the Depth Anything V2 resize.** Upstream preprocesses with
`torchvision.transforms.Resize((518, 518))` on a PIL image, which is bilinear
*with* antialias. `depth_anything_v2` here uses cv2 `INTER_LINEAR`, which is
bilinear *without*. Reducing a 3024x2268 source by four is exactly where the
two diverge most. This file writes the tensor it actually exported to
reports/inputs/hyden.npy so the difference is a measurement rather than an
assumption -- the same reason depth_pro kept its own preprocessing path.
"""

import os
import sys

# Repo root, found by walking up to the directory holding core/.
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "core")) and os.path.dirname(_R) != _R:
    _R = os.path.dirname(_R)
sys.path.insert(1, _R)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.join(CUR_DIR, "metadepth")

import numpy as np  # noqa: E402
import torch  # noqa: E402

# The clone's *parent*, not the clone. da2/__init__.py starts with
# `from ..cnn import ...` -- a relative import that leaves the package -- so
# da2 only resolves as `metadepth.da2`. Putting the clone itself on sys.path
# and importing `da2` fails with "attempted relative import beyond top-level
# package" before any weight is read.
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)


def upstream_tensor(image_path, size):
    """The input tensor upstream's own example would hand the model.

    Reproduced here rather than approximated with cv2, because the two resizes
    are different functions: torchvision's PIL path antialiases and cv2's
    INTER_LINEAR does not. Whether that difference matters is measurable, and
    the file this writes is what measures it -- but it can only measure it if
    what goes into the ONNX export is upstream's version and not ours.
    """
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(Image.open(image_path).convert("RGB")).unsqueeze(0)


def main():
    input_h, input_w = 518, 518
    encoder = "vitl"        # the only DA2 checkpoint released
    opset = 17
    onnx_sim = False

    if not os.path.isdir(CLONE):
        raise FileNotFoundError(
            f"[MDET] upstream clone missing: {CLONE}\n"
            f"       cd models/hyden && "
            f"git clone https://github.com/facebookresearch/metadepth.git")

    # The released filename is not the one upstream's snippet writes. Their
    # example says checkpoints/hyden_da2_vitl.pth; the file on HuggingFace is
    # hyden_da2_reldepth_vitl_fp32_<hash>.pth. Match on the parts that are
    # stable so a rename on download does not become a missing-file error.
    ckpt_dir = os.path.join(CLONE, "checkpoints")
    found = sorted(f for f in os.listdir(ckpt_dir)
                   if f.endswith(".pth") and "da2" in f and encoder in f
                   ) if os.path.isdir(ckpt_dir) else []
    if not found:
        raise FileNotFoundError(
            "[MDET] no HyDen-DA2 checkpoint in " + ckpt_dir + "\n"
            "       huggingface.co/facebook/hyden-da2-relative-depth is GATED.\n"
            "       Accept the FAIR Noncommercial Research License while signed\n"
            "       in, then run `huggingface-cli login` on this machine.\n"
            "       Without that the download fails with 401 GatedRepoError.")
    ckpt_path = os.path.join(ckpt_dir, found[0])
    print("[MDET] checkpoint " + found[0])

    from metadepth.da2 import HyDenDepthAnything

    print(f"[MDET] building hyden-da2 {encoder}")
    model = HyDenDepthAnything(encoder=encoder)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd.get("state_dict", sd))
    # strict=True, which is how upstream's own example loads it. The checkpoint
    # is a raw state dict, so any mismatch is a real one. A tolerance here --
    # "allow up to N missing" -- lets a partly-initialised encoder export, and
    # a partly-initialised encoder still draws a plausible depth map and still
    # produces a benchmark row. If a key genuinely has to be excused, it gets
    # named here, not counted.
    model.load_state_dict(sd, strict=True)
    model.eval()

    model_name = f"hyden_da2_{encoder}_{input_h}x{input_w}"
    out_dir = os.path.join(CUR_DIR, "onnx")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{model_name}.onnx")

    # Export on upstream's own tensor, not on randn: a real image is what the
    # constant folding sees, and it is the tensor recorded below.
    example = os.path.join(_R, "data", "example.jpg")
    dummy = (upstream_tensor(example, (input_h, input_w))
             if os.path.exists(example)
             else torch.randn(1, 3, input_h, input_w))

    print(f"[MDET] exporting -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(model, dummy, onnx_path, opset_version=opset,
                          input_names=["input"], output_names=["depth"],
                          do_constant_folding=True, dynamo=False)
    print(f"[MDET] onnx model exported "
          f"({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

    # The tensor this export was traced on, kept so verify_accuracy.py and the
    # ground-truth adapter compare against what the model was actually fed --
    # and so the torchvision-vs-cv2 resize question above has an answer on disk
    # instead of an argument.
    if os.path.exists(example):
        inputs_dir = os.path.join(_R, "reports", "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        np.save(os.path.join(inputs_dir, "hyden.npy"),
                dummy.numpy().astype(np.float32))
        print("[MDET] recorded reports/inputs/hyden.npy")

    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[MDET] onnx.checker: ok")
    for t in list(onnx_model.graph.input) + list(onnx_model.graph.output):
        dims = [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
        print(f"[MDET]   {t.name}: {dims}")

    ops = sorted({n.op_type for n in onnx_model.graph.node})
    print(f"[MDET] {len(onnx_model.graph.node)} nodes, {len(ops)} op types")

    if onnx_sim:
        from onnxsim import simplify
        simplified, ok = simplify(onnx_model)
        if not ok:
            raise RuntimeError("[MDET] simplified model failed its own check")
        onnx.save(simplified, os.path.join(out_dir, f"{model_name}_sim.onnx"))
        print("[MDET] simplified onnx saved")


if __name__ == "__main__":
    main()
