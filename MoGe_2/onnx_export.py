# by yhpark 2025-7-24
import os
os.environ['XFORMERS_DISABLED'] = '1'   # Disable xformers
import torch
import onnx
from onnxsim import simplify
from typing import *
from model_wrapper import MoGeModelWrapper
from pathlib import Path

cur_dir = os.path.dirname(os.path.abspath(__file__))
save_path = Path(cur_dir, 'onnx')
os.makedirs(save_path, exist_ok=True)


# Load model
encoder = 'vits' # 'vitl' or 'vitb', 'vits'
model_name = f"moge-2-{encoder}-normal"
checkpoint = f"{cur_dir}/MoGe/checkpoint/{model_name}/model.pt"
wrapped_model = MoGeModelWrapper.from_pretrained(checkpoint).eval()

dummy_input2 = torch.tensor(1800, dtype=torch.int64)  # Create a dummy input 1

dynamic_shape = False
if dynamic_shape : 
    dummy_input = torch.randn((1, 3, 518, 518), requires_grad=False)  # Create a dummy input 1
    export_model_path = os.path.join(save_path, f'{model_name}_dynamic.onnx')
    torch.onnx.export(
        wrapped_model, 
        (dummy_input, dummy_input2),     
        export_model_path,
        input_names=['image', 'num_tokens'],
        output_names=['points', 'normal', 'mask', 'metric_scale'],
        dynamic_axes={
            'image': {0: 'batch_size', 2: 'height', 3: 'width'},
            },
        opset_version=20
    )
else : 
    dummy_input = torch.randn((1, 3, 518, 518), requires_grad=False)  # Create a dummy input 1
    export_model_path = os.path.join(save_path, f'{model_name}_static.onnx')
    torch.onnx.export(
        wrapped_model, 
        (dummy_input, dummy_input2),     
        export_model_path,
        input_names=['image', 'num_tokens'],
        output_names=['points', 'normal', 'mask', 'metric_scale'],
        opset_version=20
    )


print(f"ONNX model exported to: {export_model_path}")

# Verify the exported ONNX model
onnx_model = onnx.load(export_model_path)
onnx.checker.check_model(export_model_path)  # Perform a validity check
print("ONNX model validation successful!")

for input in onnx_model.graph.input:
    print(f"Input: {input.name}")
    for d in input.type.tensor_type.shape.dim:
        print("dim_value:", d.dim_value, "dim_param:", d.dim_param)

for output in onnx_model.graph.output:
    print(f"Output: {output.name}")
    for d in output.type.tensor_type.shape.dim:
        print("dim_value:", d.dim_value, "dim_param:", d.dim_param)


# Simplify
try:
    model_simplified, check = simplify(onnx_model)
    if not check:
        raise RuntimeError("Simplified model is invalid.")
    if dynamic_shape : 
        export_model_sim_path = os.path.join(save_path, f'{model_name}_dynamic_sim.onnx')
    else:
        export_model_sim_path = os.path.join(save_path, f'{model_name}_static_sim.onnx')
    onnx.save(model_simplified, export_model_sim_path)
    print(f"Simplified model saved to: {export_model_sim_path}")
except Exception as e:
    print(f"Simplification failed: {e}")