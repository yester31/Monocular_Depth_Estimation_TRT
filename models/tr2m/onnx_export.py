# by yhpark 2026-8-14
# TR2M ONNX export
"""Export TR2M's per-frame pipeline as one ONNX graph.

TR2M turns relative depth into metric depth by predicting a per-pixel scale and
shift from the image *and a sentence describing it*. Four networks are
involved, but only three run on every frame:

    image ─┬─> DINOv2 ViT-L ──> patch tokens ─┐
           │                                  ├─> ScaleMap ──> A, B
    text ──┴───────── CLIP ────────────────> (only when the prompt changes)
           └─> Depth Anything ViT-S ─────────────> D_rel

    D_metric = 1 / (A * D_rel + B)

**CLIP is deliberately outside the graph.** The text embedding changes only
when the prompt does, so it stays a [1, 1, 768] input and 900 MB of CLIP
weights stay out of the engine. This script writes that embedding beside the
ONNX as text_features.npy, which is why onnx2trt.py can run in the shared
TensorRT environment without CLIP installed at all -- and why the prompt the
benchmark used is a file in the repository rather than a number in a log.

Needs the upstream clone at models/tr2m/TR2M and its weights:

    cd models/tr2m && git clone https://github.com/BeileiCui/TR2M.git
    cd TR2M && python download_weights.py

Upstream is cloned, never vendored: its pos_embed.py is CC BY-NC-SA 4.0 while
this repository is MIT.
"""

import os
import sys

# Repo root, found by walking up to the directory holding core/.
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "core")) and os.path.dirname(_R) != _R:
    _R = os.path.dirname(_R)
sys.path.insert(1, _R)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.join(CUR_DIR, "TR2M")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# The clone's modules import each other by bare name (`depth_anything.dpt`,
# `scalemap_depth`), so it has to be importable as a top-level location. CLIP
# is vendored a directory deeper -- TR2M/CLIP/clip -- so `import clip` needs
# TR2M/CLIP as well, and pip-installing OpenAI's package instead would get a
# different copy from the one upstream tested against.
for _p in (CLONE, os.path.join(CLONE, "CLIP")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

INPUT_NAMES = ["image", "text_features"]
OUTPUT_NAMES = ["metric_depth", "relative_depth", "scale_map", "shift_map"]

# Hyperparameters of the released ScaleMap head. These are not preferences --
# a mismatch loads the checkpoint into the wrong shape.
IMAGE_CHANNELS = 1024
TEXT_CHANNELS = 768
OUT_CHANNELS = 192
FEATURES = 128
CROSS_ATTN_NHEAD = 8
COEF_SCALE = 0.01
COEF_SHIFT = 0.01

DEPTH_MODEL_CONFIGS = {
    "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}


class TR2MFused(nn.Module):
    """DINOv2 + ScaleMap + Depth Anything in one traceable module.

    Mirrors the upstream per-frame path minus the text encoder. The
    interpolate guard matters: Depth Anything returns its own resolution and
    the scale maps are on the patch grid, and at 434x560 they do not always
    agree by a pixel.
    """

    def __init__(self, depth_model, image_f_model, scalemap, patch_h, patch_w):
        super().__init__()
        self.depth_model = depth_model
        self.image_f_model = image_f_model
        self.scalemap = scalemap
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)

    def forward(self, image, text_features):
        image_features = self.image_f_model.get_intermediate_layers(
            image, n=1, return_class_token=False)[0]
        scale, shift, _, _ = self.scalemap(
            image_features.float(), text_features.float(),
            patch_h=self.patch_h, patch_w=self.patch_w)
        relative = self.depth_model(image).unsqueeze(1)
        if relative.shape[-2:] != scale.shape[-2:]:
            relative = F.interpolate(relative, size=scale.shape[-2:],
                                     mode="bilinear", align_corners=True)
        return 1.0 / (scale * relative + shift), relative, scale, shift


def torch_load(path):
    """weights_only=True, with no fallback -- the rule the rest of the repo follows.

    torch.load without it unpickles arbitrary objects, so loading a checkpoint
    is running whatever its author put in the file. If one of these refuses,
    the error names the exact global that blocks it: allow that one with
    torch.serialization.add_safe_globals rather than turning the flag off.
    See tests/test_torch_load.py, which reads the source of every call site.
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def build_pipeline(depth_model, img_encoder, weight, device):
    """The three networks that run every frame. Built with the clone as cwd."""
    from depth_anything.dpt import DPT_DINOv2
    from scalemap_depth import ScaleMapModel

    encoder = "vit" + depth_model[-1]
    cfg = dict(DEPTH_MODEL_CONFIGS[encoder], localhub=True)
    print(f"[MDET] Depth Anything backbone : {depth_model} ({encoder})")
    da = DPT_DINOv2(**cfg)
    da_weight = os.path.join(CLONE, "depth_anything", f"depth_anything_{encoder}14.pth")
    if not os.path.exists(da_weight):
        raise FileNotFoundError(
            f"[MDET] {da_weight} not found. Run download_weights.py inside the clone.")
    da.load_state_dict(torch_load(da_weight))
    da = da.to(device).eval()

    print(f"[MDET] DINOv2 image encoder : {img_encoder}")
    hub_dir = os.path.join(CLONE, "torchhub", "facebookresearch_dinov2_main")
    entry = f"dinov2_{img_encoder}14"
    if os.path.isdir(hub_dir):
        dino = torch.hub.load(hub_dir, entry, source="local", pretrained=True)
    else:
        dino = torch.hub.load("facebookresearch/dinov2", entry)
    dino = dino.to(device).eval()

    print("[MDET] TR2M ScaleMap head")
    head = ScaleMapModel(IMAGE_CHANNELS, OUT_CHANNELS, TEXT_CHANNELS, FEATURES,
                         CROSS_ATTN_NHEAD, COEF_SCALE, COEF_SHIFT)
    wpath = weight if os.path.isabs(weight) else os.path.join(CLONE, weight)
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"[MDET] TR2M weight not found at {wpath}")
    ckpt = torch_load(wpath)
    head.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict)
                         and "state_dict" in ckpt else ckpt)
    return da, dino, head.to(device).eval()


def encode_prompt(text, text_encoder, device):
    """The prompt, as the [1, 1, 768] vector the graph expects.

    Written to disk so the engine build and the benchmark never need CLIP, and
    so the sentence behind a metric number is recoverable later.
    """
    import clip

    print(f"[MDET] CLIP {text_encoder} encoding the prompt")
    model, _ = clip.load(text_encoder, device=device)
    with torch.no_grad():
        feat = model.encode_text(clip.tokenize([text]).to(device))
    return feat.float().unsqueeze(1).cpu().numpy().astype(np.float32)


def main():
    input_h, input_w = 434, 560     # multiples of 14; upstream's default
    depth_model = "da_s"            # da_s | da_b | da_l
    img_encoder = "vitl"            # DINOv2 size for the ScaleMap features
    text_encoder = "ViT-L/14"
    weight = "weights/da_s_vitl_vitl.pth"
    prompt = ("The image shows a classroom with rows of desks and chairs, "
              "and blue bookshelves.")
    opset = 20
    onnx_sim = False                # untested on this graph; 1.4 GB of weights
    device = torch.device("cpu")    # tracing only; the fused graph is large

    if input_h % 14 or input_w % 14:
        raise ValueError(f"[MDET] {input_h}x{input_w} must be multiples of 14")
    if not os.path.isdir(CLONE):
        raise FileNotFoundError(
            f"[MDET] upstream clone missing: {CLONE}\n"
            f"       cd models/tr2m && git clone https://github.com/BeileiCui/TR2M.git")

    model_name = f"tr2m_{depth_model}_{img_encoder}_{input_h}x{input_w}"
    out_dir = os.path.join(CUR_DIR, "onnx")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{model_name}.onnx")
    text_path = os.path.join(out_dir, f"{model_name}_text_features.npy")

    # Upstream resolves torchhub/, weights/ and depth_anything/ against its own
    # root. Output paths are absolute above, so the chdir cannot redirect them.
    here = os.getcwd()
    os.chdir(CLONE)
    try:
        da, dino, head = build_pipeline(depth_model, img_encoder, weight, device)
        text_features = encode_prompt(prompt, text_encoder, device)
    finally:
        os.chdir(here)

    np.save(text_path, text_features)
    print(f"[MDET] prompt -> {os.path.basename(text_path)} {text_features.shape}")
    print(f'[MDET] prompt text : "{prompt}"')

    model = TR2MFused(da, dino, head, input_h // 14, input_w // 14).eval()
    dummy_image = torch.randn(1, 3, input_h, input_w, device=device)
    dummy_text = torch.from_numpy(text_features).to(device)

    print(f"[MDET] exporting -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(model, (dummy_image, dummy_text), onnx_path,
                          opset_version=opset, input_names=INPUT_NAMES,
                          output_names=OUTPUT_NAMES, dynamo=False)
    print(f"[MDET] onnx model exported "
          f"({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

    import onnx
    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_path)   # path form handles >2 GB
        print("[MDET] onnx.checker: ok")
    except Exception as e:                                    # noqa: BLE001
        print(f"[MDET] onnx.checker failed: {e}")
    for t in list(onnx_model.graph.input) + list(onnx_model.graph.output):
        dims = [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
        print(f"[MDET]   {t.name}: {dims}")

    if onnx_sim:
        from onnxsim import simplify
        print("[MDET] simplifying")
        simplified, ok = simplify(onnx_model)
        if not ok:
            raise RuntimeError("[MDET] simplified model failed its own check")
        onnx.save(simplified, os.path.join(out_dir, f"{model_name}_sim.onnx"))
        print("[MDET] simplified onnx saved")


if __name__ == "__main__":
    main()
