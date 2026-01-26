# by yhpark 2026-1-23
import os
import onnx
from onnxsim import simplify
from typing import *
import torch
from torch import nn

from depth_anything_3.api import DepthAnything3

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

class DepthAnything3OnnxWrapper(nn.Module):
    """Simplified forward that takes (B, 3, H, W) and returns depth (+ sky mask if available)."""

    def __init__(self, api_model: DepthAnything3) -> None:
        super().__init__()
        self.model = api_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # The caller is expected to validate shapes before export; keep traced graph minimal.
        model_in = image.unsqueeze(1)  # add single-view dimension

        with torch.no_grad():
            with torch.autocast(device_type=image.device.type, dtype=torch.float16):
                output =  self.model.model(
                    model_in,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=[],
                    infer_gs=False,
                    )

        depth = output["depth"]  # [B, 1, H, W] for monocular models
        sky = output["sky"]

        return depth, sky


def main ():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    # Load model
    batch_size = 1
    input_h, input_w = 518, 518 
    model = DepthAnything3.from_pretrained(f"{CUR_DIR}/Depth_Anything_V3/depth-anything/DA3METRIC-LARGE")
    model.eval()
    wrapper_model = DepthAnything3OnnxWrapper(model).to(DEVICE)

    dynamo = False    # True or False
    onnx_sim = True    # True or False
    model_name = f"DA3METRIC-LARGE"
    model_name = f"{model_name}_{input_h}x{input_w}" 
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    dummy_input = torch.randn((batch_size, 3, input_h, input_w), requires_grad=False).to(DEVICE)

    # Export the model to ONNX format
    with torch.no_grad():  # Disable gradients for efficiency
        torch.onnx.export(
            wrapper_model, 
            dummy_input,     
            export_model_path,
            input_names=['image'],
            output_names=['depth', 'sky'],
            opset_version=20,
            dynamo=dynamo,
            # export_params=True,
            # training=torch.onnx.TrainingMode.EVAL,
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