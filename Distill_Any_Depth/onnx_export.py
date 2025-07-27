# by yhpark 2025-7-20
import os
import torch
import onnx
from onnxsim import simplify

from infer import set_model as load_model

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
print(f"[MDET] using device: {DEVICE}")

class ModelWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        output, _ = self.base_model(x)  # ignore output2
        return output  # 

def main ():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Model preparation
    input_h, input_w = 518, 518 # 700, 700
    encoder = 'small' # 'large' or 'base' or 'small' or 'Large-2w-iter'
    model, _ = load_model(encoder)

    dynamo = True      # True or False
    onnx_sim = True    # True or False
    model_name = f"distill_any_depth_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    print('[MDET] Export the model to onnx format')
    input_size = (1, 3, input_h, input_w)
    dummy_input = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input

    # Export the model to ONNX format
    with torch.no_grad():  # Disable gradients for efficiency
        torch.onnx.export(
            ModelWrapper(model), 
            dummy_input, 
            export_model_path, 
            opset_version=20, 
            input_names=["input"],
            output_names=["output"],
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