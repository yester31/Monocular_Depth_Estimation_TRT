# by yhpark 2025-7-17
import os
import sys
import torch
import onnx
from onnxsim import simplify

sys.path.insert(1, os.path.join(sys.path[0], "Depth-Anything-V2"))
cur_dir = os.path.dirname(os.path.abspath(__file__))

from depth_anything_v2.dpt import DepthAnythingV2

# Load model and preprocessing transform
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# Load model
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vits' # or 'vits', 'vitb', 'vitg'
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'./Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

model_name = f"depth_anything_v2_{encoder}"
export_model_path = os.path.join(cur_dir, 'onnx', f'{model_name}.onnx')
os.makedirs(os.path.dirname(export_model_path), exist_ok=True)

# Get model input size from the model configuration
input_size = (1, 3, 518, 518)
#input_size = (1, 3, 518, 686)
dummy_input = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input

# Export the model to ONNX format
with torch.no_grad():  # Disable gradients for efficiency
    torch.onnx.export(
        model, 
        dummy_input, 
        export_model_path, 
        opset_version=20, 
        input_names=["input"],
        output_names=["output"],
        # dynamic_axes={
        #    "input": {0: "batch_size", 2: "height", 3: "width"},
        #    "output": {0: "batch_size", 1: "height", 2: "width"},
        #}
        #dynamo=True,
        #dynamic_shapes={"input": {0: "batch_size", 2: "height", 3: "width"},
        #"output": {0: "batch_size", 1: "height", 2: "width"}  # 옵션 (출력도 동적이면)
        #}
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
    export_model_sim_path = os.path.join(cur_dir, 'onnx', f'{model_name}_sim.onnx')
    onnx.save(model_simplified, export_model_sim_path)
    print(f"Simplified model saved to: {export_model_sim_path}")
except Exception as e:
    print(f"Simplification failed: {e}")