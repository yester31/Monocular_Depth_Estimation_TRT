# by yhpark 2025-7-31
import contextlib
import os
import torch
import torch.nn.functional as F
import onnx
from onnxsim import simplify
import json

from UniK3D.unik3d.models.unik3d import UniK3D
from safetensors.torch import load_file

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
print(f"[MDET] using device: {DEVICE}")

@contextlib.contextmanager
def no_antialias():
    """Export Resize nodes with antialias=0, which is all TensorRT accepts.

    `decoder.embed_rays()` resamples the camera-ray embedding with
    `flat_interpolate(..., antialias=True)`. That reaches ONNX as a Resize
    carrying antialias=1, and the TensorRT parser rejects the whole file:

        UNSUPPORTED_NODE_ATTR: (antialias == 0) && "Antialiasing is not
        supported currently."

    Measured before doing this, on data/example.jpg at 518x518, PyTorch with
    and without: the point maps are **bit-identical** -- max absolute
    difference 0.000000, correlation 1.000000, delta1 100%. PyTorch only
    antialiases when a resize shrinks an image, and at this input size that
    one does not shrink, so the flag never does anything.

    That equivalence is tied to the input size. A configuration where the ray
    resize genuinely downsamples would lose real filtering here, so re-measure
    with probe_antialias.py before changing the size.
    """
    real = F.interpolate

    def patched(*args, **kwargs):
        kwargs["antialias"] = False
        return real(*args, **kwargs)

    F.interpolate = patched
    torch.nn.functional.interpolate = patched
    try:
        yield
    finally:
        F.interpolate = real
        torch.nn.functional.interpolate = real


class UniK3DONNX(UniK3D):
    def __init__(
        self,
        config,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(config, eps)

    def forward(self, rgbs):
        B, _, H, W = rgbs.shape
        features, tokens = self.pixel_encoder(rgbs)

        inputs = {}
        inputs["image"] = rgbs
        inputs["features"] = [
            self.stacking_fn(features[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        inputs["tokens"] = [
            self.stacking_fn(tokens[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        outputs = self.pixel_decoder(inputs, [])
        outputs["rays"] = outputs["rays"].permute(0, 2, 1).reshape(B, 3, H, W)
        pts_3d = outputs["rays"] * outputs["distance"]

        return pts_3d, outputs["confidence"]

def main ():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    # Model preparation
    # 518x518 — the repo-wide comparison size. Must match onnx2trt.py,
    # which documents why no aspect-preserving variant is offered.
    # 672x896, not 518x518. These models resize internally by their own rule --
    # pad the aspect ratio into [0.5, 2.5], then scale the pixel count into
    # [200000, 600000], snapped to a multiple of 14 -- and for any 4:3 source
    # that rule lands on exactly 672x896. Feeding them a 518 square meant the
    # engine ran at a size the model would never have chosen, which is what
    # D11's 3.1x metric scale error was: not a defect in the conversion, a
    # consequence of overriding the model's own choice.
    #
    # It costs speed. 602112 pixels against 268324 is 2.24x the work, and the
    # comparison table records the input size beside every number for exactly
    # this reason.
    input_h, input_w = 672, 896
    encoder = 'vits' # 'vits' or 'vitb' or 'vitl'
    with open(f"{CUR_DIR}/UniK3D/configs/eval/{encoder}.json") as f:
        config = json.load(f)
    model = UniK3DONNX(config)
    checkpoint_path = os.path.join(CUR_DIR, "UniK3D","checkpoints", encoder, "model.safetensors")
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    model = model.eval()
    #model = model.to(DEVICE)

    dynamic = False      # False
    onnx_sim = True     # True or False
    # The dynamo exporter is what produces a working graph for this
    # model, so it is named. It used to be left to torch's default,
    # which meant the filename recorded nothing about how the ONNX was
    # produced -- and the two exporters emit different graphs.
    dynamo = True        # True or False
    model_name = f"unik3d_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    print('[MDET] Export the model to onnx format')
    input_size = (1, 3, input_h, input_w)
    dummy_input = torch.randn(input_size, requires_grad=False)  # Create a dummy input
    # dummy_input = dummy_input.to(DEVICE)  # Create a dummy input

    dynamic_axes=None
    if dynamic:
        dynamic_axes={"rgbs": {0: "batch", 2: "height", 3: "width"}}

    # Export the model to ONNX format
    with torch.no_grad(), no_antialias():  # see no_antialias() for why
        torch.onnx.export(
            model,
            dummy_input,
            export_model_path,
            opset_version=20,
            input_names=["rgbs"],
            output_names=["pts_3d", "confidence"],
            dynamic_axes=dynamic_axes,
            dynamo=dynamo,
        )

    print(f"ONNX model exported to: {export_model_path}")

    print("[MDET] Validate exported onnx model")
    try:
        onnx_model = onnx.load(export_model_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"[MDET] failed onnx.checker.check_model() : {e}")
    finally:
        onnx.checker.check_model(export_model_path)

    for input in onnx_model.graph.input:
        print(f"[MDET] Input: {input.name}")
        for d in input.type.tensor_type.shape.dim:
            print("[MDET] dim_value:", d.dim_value, "dim_param:", d.dim_param)

    for output in onnx_model.graph.output:
        print(f"[MDET] Output: {output.name}")
        for d in output.type.tensor_type.shape.dim:
            print("[MDET] dim_value:", d.dim_value, "dim_param:", d.dim_param)

    if onnx_sim :
        print("[MDET] Simplify exported onnx model")
        onnx_model = onnx.load(export_model_path)
        try:
            model_simplified, check = simplify(onnx_model)
            if not check:
                raise RuntimeError("[MDET] Simplified model is invalid.")
            
            export_model_sim_path = os.path.join(save_path, f'{model_name}_sim.onnx')
            onnx.save(model_simplified, export_model_sim_path)
            print(f"[MDET] simplified onnx model saved to: {export_model_sim_path}")
        except Exception as e:
            print(f"[MDET] simplification failed: {e}")


if __name__ == "__main__":
    main()