# by yhpark 2025-7-16
import depth_pro
import os
import torch
import onnx
from onnxsim import simplify

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
print(f"[MDET] using device: {DEVICE}")

def set_model(precision: torch.dtype = torch.float32):
    # Model Config
    CUSTOM_MONODEPTH_CONFIG_DICT = depth_pro.depth_pro.DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=f"{CUR_DIR}/ml-depth-pro/checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )

    # Load model and preprocessing transform
    model, transform = depth_pro.depth_pro.create_model_and_transforms(
        config=CUSTOM_MONODEPTH_CONFIG_DICT, 
        device=DEVICE, 
        precision=precision
        )
    model.eval()
    return model, transform

def main():
    print('[MDET] Load model')
    model, _ = set_model(precision=torch.float32)

    dynamo = True # True or False
    model_name = "depth_pro_dynamo" if dynamo else "depth_pro"
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    print('[MDET] Export the model to onnx format')
    input_size = (1, 3, model.img_size, model.img_size)
    dummy_input = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input

    with torch.no_grad():  # Disable gradients for efficiency
        torch.onnx.export(
            model, 
            dummy_input, 
            export_model_path, 
            opset_version=20, 
            input_names=["input"],
            output_names=["canonical_inverse_depth", "fov_deg"],
            dynamo=dynamo
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

    print("[MDET] Simplify exported onnx model")
    onnx_model = onnx.load(export_model_path)
    try:
        model_simplified, check = simplify(onnx_model)
        if not check:
            raise RuntimeError("Simplified model is invalid.")
        
        export_model_sim_path = os.path.join(save_path, f'{model_name}_sim.onnx')
        onnx.save(model_simplified, export_model_sim_path)
        print(f"[MDET] simplified onnx model saved to: {export_model_sim_path}")
    except Exception as e:
        print(f"[MDET] simplification failed: {e}")

if __name__ == "__main__":
    main()