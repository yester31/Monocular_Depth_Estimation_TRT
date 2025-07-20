# by yhpark 2025-7-20
import os
import sys
import torch
import onnx
from onnxsim import simplify

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(sys.path[0], "Distill-Any-Depth"))

from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from safetensors.torch import load_file 

# Helper function for model loading
def load_model_by_name(arch_name, checkpoint_path, device):
    model_kwargs = dict(
        vits=dict(
            encoder='vits', 
            features=64,
            out_channels=[48, 96, 192, 384]
        ),
        vitb=dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        vitl=dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        )
    )

    # Load model
    if arch_name == 'depthanything-large':
        model = DepthAnything(**model_kwargs['vitl']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"large/model.safetensors", repo_type="model")

    elif arch_name == 'depthanything-base':
        model = DepthAnythingV2(**model_kwargs['vitb']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"base/model.safetensors", repo_type="model")
    elif arch_name == 'depthanything-small':
        model = DepthAnythingV2(**model_kwargs['vits']).to(device)
        # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"base/model.safetensors", repo_type="model")

    else:
        raise NotImplementedError(f"Unknown architecture: {arch_name}")
    
    # safetensors 
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    del model_weights
    torch.cuda.empty_cache()
    return model

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Using device: {device}")

encoder = 'small' # 'large' or 'base', 'small'
checkpoint = f'{cur_dir}/Distill-Any-Depth/checkpoint/{encoder}/model.safetensors'
arch_name = f"depthanything-{encoder}" # 'depthanything-large', 'depthanything-base', 'depthanything-small'
model = load_model_by_name(arch_name, checkpoint, device)
model = model.eval().to(device)

model_name = f"distill_any_depth_{encoder}"
export_model_path = os.path.join(cur_dir, 'onnx', f'{model_name}.onnx')
os.makedirs(os.path.dirname(export_model_path), exist_ok=True)

# Get model input size from the model configuration
input_size = (1, 3, 518, 518)
#input_size = (1, 3, 518, 686)
dummy_input = torch.randn(input_size, requires_grad=False).to(device)  # Create a dummy input

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