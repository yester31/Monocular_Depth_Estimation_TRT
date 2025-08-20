# by yhpark 2025-8-13
import os
import torch
import onnx
from onnxsim import simplify

from omegaconf import OmegaConf
import sys
sys.path.insert(1, os.path.join(sys.path[0], "FlashDepth"))
from wrapper import FlashDepthModelWrapper

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
print(f"[MDET] using device: {DEVICE}")

def main():

    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    config_dir_path = f"{CUR_DIR}/FlashDepth/configs/flashdepth"
    cfg = OmegaConf.load(f"{config_dir_path}/config.yaml")
    model = FlashDepthModelWrapper(**dict( 
        batch_size=cfg.training.batch_size, 
        hybrid_configs=cfg.hybrid_configs,
        training=False,
        **cfg.model,
    ))

    model = model.cpu()
    checkpoint_path = f'{config_dir_path}/iter_43002.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(model_state_dict, strict=True)
    model = model.to(DEVICE)
    model.eval()
    # model = torch.jit.script(model)

    input_h, input_w = 518, 518  # divisible by 14

    dynamo = False   # True or False
    onnx_sim = False # True or False
    model_name = f"flashdepth_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    print('[MDET] Export the model to onnx format')
    input_size = (1, 3, input_h, input_w)
    dummy_input1 = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input1

    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input1, 
            export_model_path, 
            opset_version=20, 
            input_names=["image"],
            output_names=["depth"],
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
