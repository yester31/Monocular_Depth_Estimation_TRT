# by yhpark 2025-7-30
import os
os.environ['XFORMERS_DISABLED'] = '1'   # Disable xformers
import os
import torch
import onnx
from onnxsim import simplify
import math
import torch.nn as nn

import sys
sys.path.insert(1, os.path.join(sys.path[0], "Metric3D"))
from Metric3D.hubconf import *
from Metric3D.onnx.metric3d_onnx_export import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

def main ():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    input_h = 616
    input_w = 1064 
    encoder = 'vitl' # 'vits' or vitl or vitg
    if encoder == 'vits':
        model = metric3d_vit_small(pretrain=True)
    elif encoder == 'vitl':
        model = metric3d_vit_large(pretrain=True)
    elif encoder == 'vitg':
        model = metric3d_vit_giant2(pretrain=True)
    #model = metric3d_convnext_tiny(pretrain=True)
    #model = metric3d_convnext_large(pretrain=True)

    model = update_vit_sampling(model)
    model = Metric3DExportModel(model)
    model.eval()
    model.to(DEVICE)

    onnx_sim = True     # True or False
    model_name = f"metric3d_{encoder}_{input_h}x{input_w}"
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    print('[MDET] Export the model to onnx format')
    input_size = (1, 3, input_h, input_w)
    dummy_input = torch.randn(input_size, requires_grad=False)  # Create a dummy input
    dummy_input = dummy_input.to(DEVICE)  # Create a dummy input
    # torch 2.0.1
    # numpy 1.24.4
    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            export_model_path, 
            input_names=["image"],
            output_names=["pred_depth"],
            opset_version=17,
            do_constant_folding=False
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

if __name__ == "__main__":
    main()