# by yhpark 2025-7-16
import depth_pro
import os
import sys
import torch
import onnx

sys.path.insert(1, os.path.join(sys.path[0], "ml-depth-pro"))

current_directory = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessing transform
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model, transform = depth_pro.create_model_and_transforms(device=device)
model.eval()

model_name = "dinov2l16_384"
export_model_path = os.path.join(current_directory, 'onnx', f'{model_name}.onnx')
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
        )
    print(f"ONNX model exported to: {export_model_path}")

# Verify the exported ONNX model
onnx_model = onnx.load(export_model_path)
onnx.checker.check_model(export_model_path)  # Perform a validity check
print("ONNX model validation successful!")

# onnx_orig_path = os.path.join(current_directory, f'{model_name}_{device.type}_new.onnx')

# onnx.save_model(
#     onnx_model,
#     onnx_orig_path,
#     save_as_external_data=False,
#     all_tensors_to_one_file=True,
#     convert_attribute=False,
# )

# onnx.checker.check_model(onnx_orig_path)  # Perform a validity check
# print("ONNX model validation successful! 2")