# by yhpark 2026-8-14
# ZipDepth ONNX export
"""Export ZipDepth to ONNX.

The odd one out here: 6.1M parameters of reparameterised convolutions, where
every other model in this repository is a ViT or a transformer geometry
network. It is in the comparison to put a point somewhere else on the curve --
and to make TensorRT's convolution fusion path carry a row, rather than its
attention path carrying all thirteen.

Needs the upstream clone beside this file:

    cd models/zipdepth
    git clone https://github.com/fabiotosi92/ZipDepth.git
    conda create -n zipdepth -y python=3.10 && conda activate zipdepth
    cd ZipDepth && pip install -r requirements.txt && pip install -e .

The checkpoints are committed in the upstream repository, so cloning is the
whole download -- no weights step.

Size is fixed at 384x512. Upstream resizes the short side to 384 and rounds
both sides up to a multiple of 32, which for the 4:3 of data/example.jpg is
exactly 384x512. Pinning it rather than deriving it per image is the same
decision made for moge_2 and metric_anything, and for the same reason: the
engine is static and its filename carries the size, so deriving it would send a
16:9 photo looking for an engine nobody built.
"""

import os
import sys

# Repo root, found by walking up to the directory holding core/.
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "core")) and os.path.dirname(_R) != _R:
    _R = os.path.dirname(_R)
sys.path.insert(1, _R)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.join(CUR_DIR, "ZipDepth")

import torch  # noqa: E402

if CLONE not in sys.path:
    sys.path.insert(0, CLONE)


def load_state_dict(ckpt):
    """The checkpoint's weights, with any wrapper prefix removed.

    Upstream calls its own strip_state_dict_prefixes here. Reproduced rather
    than imported because that helper's module path is not something this
    repository should pin -- upstream is a clone that can move under us, and a
    two-line reimplementation cannot break in a way that is hard to see.
    `module.` comes from DataParallel and `_orig_mod.` from torch.compile;
    both are wrappers, not weights.
    """
    sd = ckpt.get("model_state_dict", ckpt)
    for prefix in ("module.", "_orig_mod."):
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def main():
    input_h, input_w = 384, 512     # short side 384, both a multiple of 32
    variant = "base"
    npu = False                     # True picks the unfold-free upsampler
    opset = 17                      # upstream's default
    onnx_sim = False

    if input_h % 32 or input_w % 32:
        raise ValueError(f"[MDET] {input_h}x{input_w} must be multiples of 32")
    if not os.path.isdir(CLONE):
        raise FileNotFoundError(
            f"[MDET] upstream clone missing: {CLONE}\n"
            f"       cd models/zipdepth && "
            f"git clone https://github.com/fabiotosi92/ZipDepth.git")

    from zipdepth.model.architecture import create_model

    ckpt_name = f"zipdepth_{variant}_npu.pth" if npu else f"zipdepth_{variant}.pth"
    ckpt_path = os.path.join(CLONE, "checkpoints", ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[MDET] checkpoint not found: {ckpt_path}")

    # upsample_unfold=False is the mobile build, which avoids the convex
    # upsampling that TensorRT handles unevenly. Both checkpoints are shipped
    # precisely so there is a fallback if Unfold turns out to be the problem.
    print(f"[MDET] building zipdepth {variant} (upsample_unfold={not npu})")
    model = create_model(variant=variant, upsample_unfold=not npu)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(load_state_dict(ckpt), strict=False)
    # strict=False is upstream's own choice; printing what it let through is
    # the difference between that being deliberate and being a silent
    # half-loaded model.
    if missing or unexpected:
        print(f"[MDET] state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected")
        for k in list(missing)[:5] + list(unexpected)[:5]:
            print(f"[MDET]   {k}")
    model.eval()

    model_name = f"zipdepth_{variant}{'_npu' if npu else ''}_{input_h}x{input_w}"
    out_dir = os.path.join(CUR_DIR, "onnx")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{model_name}.onnx")

    dummy = torch.randn(1, 3, input_h, input_w)
    print(f"[MDET] exporting -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(model, dummy, onnx_path, opset_version=opset,
                          input_names=["image"], output_names=["depth"],
                          do_constant_folding=True, dynamo=False)
    print(f"[MDET] onnx model exported "
          f"({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[MDET] onnx.checker: ok")
    for t in list(onnx_model.graph.input) + list(onnx_model.graph.output):
        dims = [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
        print(f"[MDET]   {t.name}: {dims}")

    if onnx_sim:
        from onnxsim import simplify
        simplified, ok = simplify(onnx_model)
        if not ok:
            raise RuntimeError("[MDET] simplified model failed its own check")
        onnx.save(simplified, os.path.join(out_dir, f"{model_name}_sim.onnx"))
        print("[MDET] simplified onnx saved")


if __name__ == "__main__":
    main()
