# by yhpark 2025-7-26
import os
import sys
import torch
import onnx
from onnxsim import simplify

sys.path.insert(1, os.path.join(sys.path[0], "Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
print(f"[MDET] using device: {DEVICE}")

from infer import set_model as load_model
from infer_metric import set_model as load_metric_model

def main ():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    input_h = 518 # 1036
    input_w = 518 # 1386
    encoder = 'vits'    # 'vits', 'vitb', 'vitg' 
    metric_model = True # True or False
    dataset = 'hypersim'# 'hypersim' for indoor model, 'vkitti' for outdoor model
    if metric_model:
        model, _ = load_metric_model(encoder=encoder, dataset=dataset)
    else:
        model, _ = load_model(encoder=encoder)
    dynamo = True      # True or False
    onnx_sim = True     # True or False
    dynamic = False     # Fail... (False only)
    model_name = f"depth_anything_v2_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_metric_{dataset}" if metric_model else model_name
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    print('[MDET] Export the model to onnx format')
    input_size = (1, 3, input_h, input_w)
    dummy_input = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input
    
    dynamic_axes = None 
    dynamic_shapes = None 
    if dynamic:
        if dynamo:
            dynamic_shapes = {"input": {0: "batch", 2: "height", 3: "width"}}
        else:
            dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},} 

    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            export_model_path, 
            opset_version=20, 
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            dynamo=dynamo,
            dynamic_shapes=dynamic_shapes
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