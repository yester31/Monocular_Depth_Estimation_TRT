# by yhpark 2025-7-16
import depth_pro
import os
import sys
import torch
import onnx
from onnxsim import simplify

sys.path.insert(1, os.path.join(sys.path[0], "ml-depth-pro"))
cur_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessing transform
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model, transform = depth_pro.create_model_and_transforms(device=device)
model.eval()

model_name = "dinov2l16_384"
export_model_path = os.path.join(cur_dir, 'onnx', f'{model_name}.onnx')
os.makedirs(os.path.dirname(export_model_path), exist_ok=True)

# Get model input size from the model configuration
input_size = (1, 3, model.img_size, model.img_size)
dummy_input = torch.randn(input_size, requires_grad=False).to(device)  # Create a dummy input

# Export the model to ONNX format
with torch.no_grad():  # Disable gradients for efficiency
    torch.onnx.export(
        model, 
        dummy_input, 
        export_model_path, 
        opset_version=20, 
        input_names=["input"],
        output_names=["canonical_inverse_depth", "fov_deg"],
        dynamo=True
        )
print(f"ONNX model exported to: {export_model_path}")

# Validation
onnx_model = onnx.load(export_model_path)
onnx.checker.check_model(onnx_model) 
print("ONNX model validation successful!")

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