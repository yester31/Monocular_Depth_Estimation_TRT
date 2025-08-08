# by yhpark 2025-8-9
import os
import torch
import onnx
from onnxsim import simplify

from wrapper import MeFlowModelWrapper

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
print(f"[MDET] using device: {DEVICE}")

def main():

    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    model = MeFlowModelWrapper(downsample_factor=8,
                feature_channels=128,
                corr_radius=4,
                hidden_dim=128,
                context_dim=128,
                mixed_precision=False)

    checkpoint = torch.load(f'{CUR_DIR}/MeFlow/pretrained_models/sintel.pth')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=False)
    model.to(DEVICE)
    model.eval()

    input_h, input_w = 288, 512  # divisible by 8

    dynamo = False   # True or False
    onnx_sim = False # True or False
    model_name = f"MeFlow_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    print('[MDET] Export the model to onnx format')
    input_size = (1, 3, input_h, input_w)
    dummy_input1 = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input1
    dummy_input2 = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input2
    iters = 20

    with torch.no_grad():
        torch.onnx.export(
            model, 
            (dummy_input1, dummy_input2, iters), 
            export_model_path, 
            opset_version=20, 
            input_names=["image1", "image2"],
            output_names=["flow_low", "flow_up"],
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

if __name__ == '__main__':
    main()
