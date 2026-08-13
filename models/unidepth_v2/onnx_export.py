# by yhpark 2025-7-29
import os
import torch
import onnx
from onnxsim import simplify
import json

import sys
# Repo root, found by walking up to the directory holding core/ rather
# than counting "..". Counting breaks the moment a script moves, and
# this one has moved: Phase 4 put every model under models/.
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "core")) and os.path.dirname(_R) != _R:
    _R = os.path.dirname(_R)
sys.path.insert(1, _R)  # repo root, for core
from core.onnx_tools import demote_float64

import huggingface_hub
from UniDepth.unidepth.models.unidepthv2.export import UniDepthV2ONNX

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

def main ():
    print('[MDET] Load model')
    save_path = os.path.join(CUR_DIR, 'onnx')
    os.makedirs(save_path, exist_ok=True)

    # 518x518 — the repo-wide comparison size. Must match onnx2trt.py,
    # which documents why no aspect-preserving variant is offered.
    # 672x896, not 518x518. These models resize internally by their own rule --
    # pad the aspect ratio into [0.5, 2.5], then scale the pixel count into
    # [200000, 600000], snapped to a multiple of 14 -- and for any 4:3 source
    # that rule lands on exactly 672x896. Feeding them a 518 square meant the
    # engine ran at a size the model would never have chosen, which is what
    # D11's 3.1x metric scale error was: not a defect in the conversion, a
    # consequence of overriding the model's own choice.
    #
    # It costs speed. 602112 pixels against 268324 is 2.24x the work, and the
    # comparison table records the input size beside every number for exactly
    # this reason.
    input_h, input_w = 672, 896
    encoder = 'vits' # 'vits' or vitb or vitl
    with open(os.path.join(f"{CUR_DIR}/UniDepth/configs", f"config_v2_{encoder}14.json")) as f:
        config = json.load(f)
        config["training"]["export"] = True

    model = UniDepthV2ONNX(config)
    path = huggingface_hub.hf_hub_download(
        repo_id=f"lpiccinelli/unidepth-v2-{encoder}14",
        filename=f"pytorch_model.bin",
        repo_type="model",
    )
    info = model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=False)
    model = model.eval().to(DEVICE) 

    dynamo = False      # False
    onnx_sim = True     # True or False
    model_name = f"uni_depth_v2_{encoder}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamo" if dynamo else model_name
    export_model_path = os.path.join(save_path, f'{model_name}.onnx')

    print('[MDET] Export the model to onnx format')
    input_size = (1, 3, input_h, input_w)
    dummy_input = torch.randn(input_size, requires_grad=False).to(DEVICE)  # Create a dummy input
    
    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            export_model_path, 
            opset_version=20, 
            input_names=["rgbs"],
            output_names=["pts_3d", "confidence", "intrinsics"],
            dynamo=dynamo,
        )
        print(f"[MDET] onnx model exported to: {export_model_path}")

    # The decoder builds coordinate grids in float64 and casts back, so the
    # exported graph mixes float32 and float64 in one operator and onnxruntime
    # refuses to load it:
    #
    #   Type Error: Type parameter (T) of Optype (LayerNormalization) bound to
    #   different types (tensor(double) and tensor(float))
    #
    # Rewritten rather than patched at the source: there are eleven such sites
    # across the decoder, against VGGT's one. TensorRT has no fp64 and was
    # already computing these in fp32, so the engine is unchanged; the file now
    # says what the engine does, which is what verify_accuracy.py needs to read
    # it. See core/onnx_tools.py.
    demote_float64(export_model_path)

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