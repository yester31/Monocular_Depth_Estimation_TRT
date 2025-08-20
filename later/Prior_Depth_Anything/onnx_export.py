# by yhpark 2025-8-1
import os
import torch
import onnx
from onnxsim import simplify

import sys
sys.path.insert(1, os.path.join(sys.path[0], "Prior-Depth-Anything"))
from vggt.models.vggt import VGGT


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")   

### NOTICE ###
# Before exporting to onnx, edit line 55 in Prior-Depth-Anything/vggt/layers/rope.py.
'''
    # Comment out line 55
    # positions = torch.cartesian_prod(y_coords, x_coords) # <- original 55 line
    # Add the three lines below
    yy = y_coords.unsqueeze(1).expand(-1, x_coords.size(0))  # [H, W]
    xx = x_coords.unsqueeze(0).expand(y_coords.size(0), -1)  # [H, W]
    positions = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # [H*W, 2] 
'''
class VGGTDepthOnlyWrapper(VGGT):
    def __init__(self):
        super().__init__()

    def forward(self, images): # images [1, S, 3, H, W]
        # Aggregate features using the backbone
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=False):
            # Camera pose estimation (if enabled)
            #if self.camera_head is not None:
            #    pose_enc_list = self.camera_head(aggregated_tokens_list)
            #    pose_enc = pose_enc_list[-1]  # Use the last iteration output

            # Depth and confidence map prediction (if enabled)
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

            # 3D world point prediction and tracking are disabled for ONNX export
            # Uncomment the following lines if needed for additional outputs
            # if self.point_head is not None:
            #     pts3d, pts3d_conf = self.point_head(
            #         aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            #     )
            #     world_points = pts3d
            #     world_points_conf = pts3d_conf

        #return pose_enc, depth, depth_conf
        return depth

def main ():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)
    dtype = torch.half # torch.half or torch.float32

    # Model preparation
    input_h, input_w = 518, 518
    vggt = VGGTDepthOnlyWrapper().from_pretrained("facebook/VGGT-1B").to(DEVICE).eval()


    dynamic = False    # False
    dynamo = False     # False
    onnx_sim = False    # False
    model_name = f"vggt_only_depth_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    model_name = f"{model_name}_fp16" if dtype == torch.half else model_name    
    export_model_path = os.path.join(save_path, model_name, f'{model_name}.onnx')
    os.makedirs(os.path.join(save_path, model_name), exist_ok=True)

    print('[MDET] Export the model to onnx format')
    dummy_input = torch.randn((1, 1, 3, input_h, input_w), requires_grad=False).to(DEVICE)  # Create a dummy input

    dynamic_axes = None 
    dynamic_shapes = None 
    if dynamic:
        if dynamo:
            dynamic_shapes={"images": {0: "batch"},} 
        else:
            dynamic_axes = {
                "images": {0: "batch"}, 
                # "pose_enc": {0: "batch"},
                "depth": {0: "batch"},
                # "depth_conf": {0: "batch"},
                }
    # Export the model to ONNX format
    with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype):
        with torch.no_grad():  # Disable gradients for efficiency
            torch.onnx.export(
                model, 
                dummy_input, 
                export_model_path, 
                opset_version=17, 
                input_names=["images"],
                #output_names=["pose_enc", "depth", "depth_conf"],
                output_names=["depth"],
                dynamic_axes=dynamic_axes, 
                dynamo=dynamo,
                dynamic_shapes=dynamic_shapes,
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
    main()