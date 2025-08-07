import sys
import argparse
import os
import torch
import onnx
from onnxsim import simplify

sys.path.insert(1, os.path.join(sys.path[0], "RAFT", "core"))
from RAFT.core.raft import RAFT

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
print(f"[MDET] using device: {DEVICE}")

class RAFTModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model  

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, iters=20):
        
        flow_predictions = self.model(image1, image2, iters)

        return flow_predictions[-1]


def main():

    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=f'{CUR_DIR}/RAFT/models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--path', default=f'{CUR_DIR}/../Monocular_Depth_Estimation_TRT/video_frames', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    input_h, input_w = 288, 512  # divisible by 8

    dynamic = False # Fail... (False only)
    dynamo = False   # True or False
    onnx_sim = False # True or False
    model_name = f"raft_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    print('[MDET] Export the model to onnx format')
    input_size = (1, 3, input_h, input_w)
    dummy_input1 = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input1
    dummy_input2 = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input2

    dynamic_axes = None 
    dynamic_shapes = None 
    if dynamic:
        if dynamo:
            dynamic_shapes = {"image1": {0: "batch", 2: "height", 3: "width"},"image2": {0: "batch", 2: "height", 3: "width"}}
        else:
            dynamic_axes={"image1": {0: "batch", 2: "height", 3: "width"},"image2": {0: "batch", 2: "height", 3: "width"}} 

    with torch.no_grad():
        torch.onnx.export(
            RAFTModelWrapper(model), 
            (dummy_input1, dummy_input2, 20), 
            export_model_path, 
            opset_version=20, 
            input_names=["image1", "image2"],
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

if __name__ == '__main__':
    main()
