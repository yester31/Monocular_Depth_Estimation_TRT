# by yhpark 2025-8-5
#
# Split export: VGGT as three engines instead of one.
#
#   aggregator   images                 -> aggregated_tokens_list
#   depth_head   aggregated_tokens_list -> depth, depth_conf
#   camera_head  aggregated_tokens_list -> pose_enc
#
# onnx_export.py is the single-engine alternative and emits depth only. This
# path gives confidence and camera pose as well, and lets the heavy aggregator
# run once while heads are swapped. Both work; see VGGT/README.md.
# (Was onnx_export2.py — the numeric suffix read like a revision.)
import os
import sys
# The clone root must be on sys.path, not just importable as vggt.vggt:
# upstream's models/vggt.py imports its siblings absolutely
# (`from vggt.models.aggregator import Aggregator`), so without this the
# import below fails with "No module named 'vggt.models'".
sys.path.insert(1, os.path.join(sys.path[0], "vggt"))
# Repo root, for core.export_compat. Found by walking up to the directory
# holding core/ rather than counting "..".
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "core")) and os.path.dirname(_R) != _R:
    _R = os.path.dirname(_R)
sys.path.insert(1, _R)
import torch
import onnx
from onnxsim import simplify

from vggt.vggt.models.vggt import VGGT
from core.export_compat import (no_cartesian_prod,
                                float32_sincos_pos_embed)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

# The `### NOTICE ###` that stood here told the reader to edit
# vggt/vggt/layers/rope.py by hand before exporting, replacing
# torch.cartesian_prod with an expand-and-stack, because the TorchScript
# exporter has no symbolic for aten::cartesian_prod.
#
# onnx_export.py had the same notice and it was replaced by
# core.export_compat.no_cartesian_prod under D17 -- but only there, so this
# path still asked for a hand-edit that nothing checks, that a git pull in the
# clone undoes, and whose absence fails partway through the export with
# UnsupportedOperatorError. Both exports use the same patch now.

# [1,1,3,518,518] -> Aggregator -> [24, 1, 1, 1374, 2048]
#                                                         -> depth_head -> [1,1,518,518], [1,1,518,518]
#                                                         -> camera_head -> [1,1,9]

class VGGT_Aggregator_Wrapper(VGGT):
    def __init__(self):
        super().__init__()

    def forward(self, images): # images [1, S, 3, H, W]
        # Aggregate features using the backbone
        aggregated_tokens_list, patch_start_idx = self.aggregator(images) # patch_start_idx = 5
        aggregated_tokens_list = torch.stack(aggregated_tokens_list)
        return aggregated_tokens_list

class VGGT_depth_head_Wrapper(VGGT):
    def __init__(self):
        super().__init__()

    def forward(self, aggregated_tokens_list, images): # aggregated_tokens_list [23, 1, 1, 1374, 2048]
        aggregated_tokens_list = list(aggregated_tokens_list)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=False):
            # Depth and confidence map prediction (if enabled)
            depth, depth_conf = self.depth_head(aggregated_tokens_list, images, patch_start_idx=5)

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


def load_vggt_weights():
    """Prefer vggt/checkpoints/model.pt, fall back to the HuggingFace URL.

    from_pretrained("facebook/VGGT-1B") stood here, which is a 4.7 GB download
    each time the cache is cold -- and the cache is cold on this machine, while
    the checkpoint has been sitting in vggt/checkpoints all along, because
    infer.py and onnx_export.py read it from there. Same reason as the notice
    above: the single-engine path was fixed and this one was not.
    """
    local = os.path.join(CUR_DIR, "vggt", "checkpoints", "model.pt")
    if os.path.isfile(local):
        print(f"[MDET] weights: {local}")
        return torch.load(local, map_location="cpu", weights_only=True)
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    print(f"[MDET] weights: {url} (no local copy; ~4.7 GB)")
    return torch.hub.load_state_dict_from_url(url, weights_only=True)


def export_aggregator():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)
    dtype = torch.float32 # torch.half or torch.float32

    # Model preparation
    input_h, input_w = 518, 518
    model = VGGT_Aggregator_Wrapper()
    model.load_state_dict(load_vggt_weights())
    model.to(DEVICE).eval()

    dynamic = False    # False
    onnx_sim = False    # False
    model_name = f"vggt_aggregator_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_fp16" if dtype == torch.half else model_name    
    export_model_path = os.path.join(save_path, model_name, f'{model_name}.onnx')
    os.makedirs(os.path.join(save_path, model_name), exist_ok=True)

    print('[MDET] Export the model to onnx format')
    images = torch.randn((1, 1, 3, input_h, input_w), requires_grad=False).to(DEVICE)  # Create a dummy input

    dynamic_axes = None 
    if dynamic:
        dynamic_axes = {"images": {0: "batch"},}
    # Export the model to ONNX format
    with (torch.amp.autocast(device_type=DEVICE.type, dtype=dtype),
          no_cartesian_prod(),
          float32_sincos_pos_embed()):
        with torch.no_grad():  # Disable gradients for efficiency
            torch.onnx.export(
                model, 
                images, 
                export_model_path, 
                opset_version=21, 
                input_names=["images"],
                output_names=["aggregated_tokens_list"],
                dynamo=False,
                dynamic_axes=dynamic_axes, 
            )

    print(f"ONNX model exported to: {export_model_path}")
    torch.cuda.empty_cache()

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
    model = VGGT_depth_head_Wrapper()
    model.load_state_dict(load_vggt_weights())
    model.to(DEVICE).eval()

    dynamic = False    # False
    onnx_sim = False    # False
    model_name = f"vggt_depth_head_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_fp16" if dtype == torch.half else model_name    
    export_model_path = os.path.join(save_path, model_name, f'{model_name}.onnx')
    os.makedirs(os.path.join(save_path, model_name), exist_ok=True)

    print('[MDET] Export the model to onnx format')
    aggregated_tokens_list = torch.randn((24, 1, 1, 1374, 2048), requires_grad=False).to(DEVICE)  # Create a dummy input
    images = torch.randn((1, 1, 3, input_h, input_w), requires_grad=False).to(DEVICE)  # Create a dummy input

    dynamic_axes = None 
    if dynamic:
        dynamic_axes = {"images": {0: "batch"},}
    # Export the model to ONNX format
    with (torch.amp.autocast(device_type=DEVICE.type, dtype=dtype),
          no_cartesian_prod(),
          float32_sincos_pos_embed()):
        with torch.no_grad():  # Disable gradients for efficiency
            torch.onnx.export(
                model, 
                (aggregated_tokens_list, images), 
                export_model_path, 
                opset_version=21, 
                input_names=["aggregated_tokens_list", "images"],
                output_names=["depth", "depth_conf"],
                dynamo=False,
                dynamic_axes=dynamic_axes, 
            )

    print(f"ONNX model exported to: {export_model_path}")
    torch.cuda.empty_cache()

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
    model = VGGT_camera_head_Wrapper()
    model.load_state_dict(load_vggt_weights())
    model.to(DEVICE).eval()

    dynamic = False    # False
    onnx_sim = False    # False
    model_name = f"vggt_camera_head_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_fp16" if dtype == torch.half else model_name    
    export_model_path = os.path.join(save_path, model_name, f'{model_name}.onnx')
    os.makedirs(os.path.join(save_path, model_name), exist_ok=True)

    print('[MDET] Export the model to onnx format')
    aggregated_tokens_list = torch.randn((24, 1, 1, 1374, 2048), requires_grad=False).to(DEVICE)  # Create a dummy input

    dynamic_axes = None 
    if dynamic:
        dynamic_axes = {"images": {0: "batch"},}
    # Export the model to ONNX format
    with (torch.amp.autocast(device_type=DEVICE.type, dtype=dtype),
          no_cartesian_prod(),
          float32_sincos_pos_embed()):
        with torch.no_grad():  # Disable gradients for efficiency
            torch.onnx.export(
                model, 
                (aggregated_tokens_list), 
                export_model_path, 
                opset_version=21, 
                input_names=["aggregated_tokens_list"],
                output_names=["pose_enc"],
                dynamo=False,
                dynamic_axes=dynamic_axes, 
            )

    print(f"ONNX model exported to: {export_model_path}")
    torch.cuda.empty_cache()

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
    # All three, because the split is only usable as three. Two of these were
    # commented out, so a plain run produced one of the parts and no error --
    # which is why the split has never been timed against the single engine
    # despite both scripts existing since August.
    #
    # Each function builds its own VGGT and calls torch.cuda.empty_cache()
    # before returning, so they run one after another rather than together.
    export_aggregator()
    export_depth_head()
    export_camera_head()