# by yhpark 2025-8-5
import os
import torch
import onnx
from onnxsim import simplify

from vggt.vggt.models.vggt import VGGT

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")   

### NOTICE ###
# Before exporting to onnx, edit line 55 in vggt/vggt/layers/rope.py.
'''
    # Comment out line 55
    # positions = torch.cartesian_prod(y_coords, x_coords) # <- original 55 line
    # Add the three lines below
    yy = y_coords.unsqueeze(1).expand(-1, x_coords.size(0))  # [H, W]
    xx = x_coords.unsqueeze(0).expand(y_coords.size(0), -1)  # [H, W]
    positions = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # [H*W, 2] 
'''

# [1,1,3,518,518] -> Aggregator -> [24, 1, 1, 1374, 2048]
#                                                         -> depth_head -> [1,1,518,518], [1,1,518,518]
#                                                         -> camera_head -> [1,1,9]

class VGGT_Aggregator_Wrapper(VGGT):
    def __init__(self):
        super().__init__()

    def forward(self, images): # images [1, S, 3, H, W]
        # Aggregate features using the backbone
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        aggregated_tokens_list = torch.stack(aggregated_tokens_list)
        return aggregated_tokens_list

class VGGT_depth_head_Wrapper(VGGT):
    def __init__(self):
        super().__init__()

    def forward(self, images, aggregated_tokens_list): # aggregated_tokens_list [23, 1, 1, 1374, 2048]
        aggregated_tokens_list = list(aggregated_tokens_list)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=False):
            # Depth and confidence map prediction (if enabled)
            depth, depth_conf = self.depth_head(aggregated_tokens_list, images=images, patch_start_idx=5)

        return depth, depth_conf

class VGGT_camera_head_Wrapper(VGGT):
    def __init__(self):
        super().__init__()

    def forward(self, aggregated_tokens_list): # aggregated_tokens_list [23, 1, 1, 1374, 2048]
        aggregated_tokens_list = list(aggregated_tokens_list)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=False):
            # Camera pose estimation (if enabled)
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            pose_enc = pose_enc_list[-1]  # Use the last iteration output

        return pose_enc

def export_aggregator():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)
    dtype = torch.float32 # torch.half or torch.float32

    # Model preparation
    input_h, input_w = 518, 518
    model = VGGT_Aggregator_Wrapper().from_pretrained("facebook/VGGT-1B").to(DEVICE).eval()

    dynamic = False    # False
    onnx_sim = False    # False
    model_name = f"vggt_aggregator_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_fp16" if dtype == torch.half else model_name    
    export_model_path = os.path.join(save_path, model_name, f'{model_name}.onnx')
    os.makedirs(os.path.join(save_path, model_name), exist_ok=True)

    print('[MDET] Export the model to onnx format')
    dummy_input = torch.randn((1, 1, 3, input_h, input_w), requires_grad=False).to(DEVICE)  # Create a dummy input

    dynamic_axes = None 
    if dynamic:
        dynamic_axes = {"images": {0: "batch"},}
    # Export the model to ONNX format
    with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype):
        with torch.no_grad():  # Disable gradients for efficiency
            torch.onnx.export(
                model, 
                dummy_input, 
                export_model_path, 
                opset_version=17, 
                input_names=["images"],
                output_names=["aggregated_tokens_list"],
                dynamic_axes=dynamic_axes, 
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
            
            export_model_sim_path = os.path.join(save_path, f'{model_name}_sim', f'{model_name}_sim.onnx')
            os.makedirs(os.path.join(save_path, f'{model_name}_sim'), exist_ok=True)
            onnx.save(model_simplified, export_model_sim_path)
            print(f"[MDET] simplified onnx model saved to: {export_model_sim_path}")
        except Exception as e:
            print(f"[MDET] simplification failed: {e}")

def export_depth_head():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)
    dtype = torch.half # torch.half or torch.float32

    # Model preparation
    input_h, input_w = 518, 518
    model = VGGT_depth_head_Wrapper().from_pretrained("facebook/VGGT-1B").to(DEVICE).eval()

    dynamic = False    # False
    onnx_sim = False    # False
    model_name = f"vggt_depth_head_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_fp16" if dtype == torch.half else model_name    
    export_model_path = os.path.join(save_path, model_name, f'{model_name}.onnx')
    os.makedirs(os.path.join(save_path, model_name), exist_ok=True)

    print('[MDET] Export the model to onnx format')
    dummy_input = torch.randn((1, 1, 3, input_h, input_w), requires_grad=False).to(DEVICE)  # Create a dummy input
    dummy_input2 = torch.randn((24, 1, 1, 1374, 2048), requires_grad=False).to(DEVICE)  # Create a dummy input

    dynamic_axes = None 
    if dynamic:
        dynamic_axes = {"images": {0: "batch"},}
    # Export the model to ONNX format
    with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype):
        with torch.no_grad():  # Disable gradients for efficiency
            torch.onnx.export(
                model, 
                (dummy_input, dummy_input2), 
                export_model_path, 
                opset_version=17, 
                input_names=["images", "aggregated_tokens_list"],
                output_names=["depth", "depth_conf"],
                dynamic_axes=dynamic_axes, 
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
            
            export_model_sim_path = os.path.join(save_path, f'{model_name}_sim', f'{model_name}_sim.onnx')
            os.makedirs(os.path.join(save_path, f'{model_name}_sim'), exist_ok=True)
            onnx.save(model_simplified, export_model_sim_path)
            print(f"[MDET] simplified onnx model saved to: {export_model_sim_path}")
        except Exception as e:
            print(f"[MDET] simplification failed: {e}")

def export_camera_head():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)
    dtype = torch.half # torch.half or torch.float32

    # Model preparation
    input_h, input_w = 518, 518
    model = VGGT_camera_head_Wrapper().from_pretrained("facebook/VGGT-1B").to(DEVICE).eval()

    dynamic = False    # False
    onnx_sim = False    # False
    model_name = f"vggt_camera_head_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_fp16" if dtype == torch.half else model_name    
    export_model_path = os.path.join(save_path, model_name, f'{model_name}.onnx')
    os.makedirs(os.path.join(save_path, model_name), exist_ok=True)

    print('[MDET] Export the model to onnx format')
    dummy_input = torch.randn((24, 1, 1, 1374, 2048), requires_grad=False).to(DEVICE)  # Create a dummy input

    dynamic_axes = None 
    if dynamic:
        dynamic_axes = {"images": {0: "batch"},}
    # Export the model to ONNX format
    with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype):
        with torch.no_grad():  # Disable gradients for efficiency
            torch.onnx.export(
                model, 
                (dummy_input), 
                export_model_path, 
                opset_version=17, 
                input_names=["aggregated_tokens_list"],
                output_names=["pose_enc"],
                dynamic_axes=dynamic_axes, 
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
            
            export_model_sim_path = os.path.join(save_path, f'{model_name}_sim', f'{model_name}_sim.onnx')
            os.makedirs(os.path.join(save_path, f'{model_name}_sim'), exist_ok=True)
            onnx.save(model_simplified, export_model_sim_path)
            print(f"[MDET] simplified onnx model saved to: {export_model_sim_path}")
        except Exception as e:
            print(f"[MDET] simplification failed: {e}")

if __name__ == "__main__":
    export_aggregator()
    export_depth_head()
    export_camera_head()