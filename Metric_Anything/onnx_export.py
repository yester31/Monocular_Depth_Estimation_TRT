# by yhpark 2025-7-26
import os
import sys
import torch
import onnx
from onnxsim import simplify
from typing import *
import torch.nn.functional as F

sys.path.insert(1, os.path.join(sys.path[0], "metric_anything/models/student_pointmap"))
from moge.model.v2 import MoGeModel

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print(f"[MDET] using device: {DEVICE}")


class MoGeModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model  # 내부에 모델 보관
        self.num_tokens = torch.tensor(3600, dtype=torch.int32).to(DEVICE)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        original_interpolate = F.interpolate

        def safe_interpolate(*args, **kwargs):
            kwargs["antialias"] = False
            return original_interpolate(*args, **kwargs)

        F.interpolate = safe_interpolate
        try:

            out = self.model(image, self.num_tokens)
        finally:
            F.interpolate = original_interpolate

        return out


def main():
    print("[MDET] Load model")
    save_path = os.path.join(CUR_DIR, "onnx")
    os.makedirs(save_path, exist_ok=True)

    input_h = 518  # 1036
    input_w = 518  # 1386

    checkpoint = f"{CUR_DIR}/checkpoints/student_pointmap.pt"
    model = MoGeModel.from_pretrained(checkpoint).to(DEVICE)
    model.eval()
    wrapped_model = MoGeModelWrapper(model)

    dynamo = False  # True or False
    onnx_sim = True  # True or False
    model_name = f"metric_anything_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    export_model_path = os.path.join(save_path, f"{model_name}.onnx")

    print("[MDET] Export the model to onnx format")
    input_size = (1, 3, input_h, input_w)
    dummy_input = torch.randn(input_size, requires_grad=False).to(
        DEVICE
    )  # Create a dummy input

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            export_model_path,
            opset_version=20,
            input_names=["image"],
            output_names=["points", "mask", "metric_scale"],
            dynamo=dynamo,
        )
        print(f"[MDET] onnx model exported to: {export_model_path}")

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

    if onnx_sim:
        print("[MDET] Simplify exported onnx model")
        onnx_model = onnx.load(export_model_path)
        try:
            model_simplified, check = simplify(onnx_model)
            if not check:
                raise RuntimeError("[MDET] Simplified model is invalid.")

            export_model_sim_path = os.path.join(save_path, f"{model_name}_sim.onnx")
            onnx.save(model_simplified, export_model_sim_path)
            print(f"[MDET] simplified onnx model saved to: {export_model_sim_path}")
        except Exception as e:
            print(f"[MDET] simplification failed: {e}")


if __name__ == "__main__":
    main()
