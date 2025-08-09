import sys
import argparse
import os
import torch
import torch.nn as nn

import onnx
from onnxsim import simplify

sys.path.insert(1, os.path.join(sys.path[0], "NeuFlow_v2"))
from NeuFlow_v2.NeuFlow.neuflow import NeuFlow

from infer import *
from wrapper import NeuFlowModelWrapper

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
print(f"[MDET] using device: {DEVICE}")

def main():

    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    model = NeuFlowModelWrapper().to(DEVICE)
    checkpoint = torch.load(f'{CUR_DIR}/NeuFlow_v2/neuflow_mixed.pth', map_location='cuda')
    model.load_state_dict(checkpoint['model'], strict=True)
    '''
    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
            delattr(m, "norm1")  # remove batchnorm
            delattr(m, "norm2")  # remove batchnorm
            m.forward = m.forward_fuse  # update forward
    '''
    model.eval()
    #model.half()

    input_h, input_w = 288, 512  # divisible by 16
    # input_h, input_w = 432, 768

    model.init_bhwd(1, input_h, input_w, 'cuda', False)

    dynamo = True   # True or False
    onnx_sim = True # True or False
    model_name = f"neuflow_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    print('[MDET] Export the model to onnx format')
    input_size = (1, 3, input_h, input_w)
    dummy_input1 = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input
    dummy_input2 = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input

    with torch.no_grad():
        torch.onnx.export(
            model, 
            #(dummy_input1.half(), dummy_input2.half()), 
            (dummy_input1, dummy_input2), 
            export_model_path, 
            opset_version=20, 
            input_names=["image1", "image2"],
            output_names=["flow"],
            dynamo=dynamo,
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
