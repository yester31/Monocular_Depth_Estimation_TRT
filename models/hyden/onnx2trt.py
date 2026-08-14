# by yhpark 2026-8-14
# HyDen (MetaDepth) TensorRT model generation
"""Build the HyDen-DA2 engine, benchmark it, and write a depth image.

Two things about this model are unsettled on purpose, and this file keeps them
that way rather than guessing:

**The input tensor comes from reports/inputs/hyden.npy, not from
core/preprocess.py.** Upstream resizes with torchvision on a PIL image, which
antialiases; every stretch model here uses cv2 INTER_LINEAR, which does not.
Until those two are compared on the same source, feeding this engine the cv2
version would be measuring a model nobody released. onnx_export.py writes that
file from upstream's own transform, so the two paths cannot drift apart
silently.

**The output orientation is unmeasured.** Depth Anything V2's relative
checkpoint emits disparity, but HyDen replaces the decoder and this repository
does not inherit orientation by family resemblance -- zipdepth's was measured
(-0.892 against true depth) before it was written down. This file prints the
statistic and refuses to name the output until someone has looked at it.
"""

import os
import sys

# Repo root, found by walking up to the directory holding core/.
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "core")) and os.path.dirname(_R) != _R:
    _R = os.path.dirname(_R)
sys.path.insert(1, _R)

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-GUI mode
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from core import common  # noqa: E402
from core.common import *  # noqa: E402,F401,F403
from core import bench  # noqa: E402

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    iteration = 100
    warmup = 20
    precision = "fp16"              # 'fp32' or 'fp16'
    input_h, input_w = 518, 518
    encoder = "vitl"

    model_name = f"hyden_da2_{encoder}_{input_h}x{input_w}"
    onnx_model_path = os.path.join(CUR_DIR, "onnx", f"{model_name}.onnx")
    engine_file_path = os.path.join(CUR_DIR, "engine", f"{model_name}_{precision}.engine")
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    save_dir_path = os.path.join(CUR_DIR, "results")
    os.makedirs(save_dir_path, exist_ok=True)

    image_file_name = "example.jpg"
    image_path = os.path.join(_R, "data", image_file_name)
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise FileNotFoundError(f"[MDET] could not read {image_path}")
    ori_shape = raw_image.shape[:2]
    print(f"[MDET] original image size : {ori_shape}")

    # Upstream's tensor, recorded by onnx_export.py. Not rebuilt here: see the
    # module docstring. If it is missing, the export has not run, and building
    # an engine on a cv2 approximation would produce a row that looks fine and
    # measures the wrong preprocessing.
    oracle = os.path.join(_R, "reports", "inputs", "hyden.npy")
    if not os.path.exists(oracle):
        raise FileNotFoundError(
            f"[MDET] {os.path.relpath(oracle, _R)} missing -- run onnx_export.py "
            f"first. It records upstream's own preprocessing, which is not the "
            f"cv2 stretch the other 518x518 models use.")
    batch_images = np.ascontiguousarray(np.load(oracle).astype(np.float32))
    print(f"[MDET] after preprocess shape : {batch_images.shape}")

    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:

        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = batch_images

        def run():
            return common.do_inference(context, engine=engine, bindings=bindings,
                                       inputs=inputs, outputs=outputs, stream=stream)

        trt_outputs, samples = bench.measure(run, warmup=warmup, iterations=iteration)
        pred = np.array(trt_outputs[0]).reshape(input_h, input_w)

        bench.record("hyden", samples, encoder=encoder, warmup=warmup,
                     precision=precision, profile="bench",
                     input_h=input_h, input_w=input_w,
                     engine_path=engine_file_path,
                     outputs={"prediction": pred},
                     model_input=batch_images)
        print(f"[MDET] prediction : min {pred.min():.5f} , max {pred.max():.5f}")
        print("[MDET] output_form is 'unmeasured' in spec.json. Settle it with "
              "core.gt.orientation against DIODE before this row is scored -- "
              "a wrong guess flips near and far and the scale+shift fit hides it.")

        # ===================================================================
        print("[MDET] Generate color depth image")
        # Normalised by its own range because the value has no unit. Drawn as
        # it comes out: applying 1/x here would bake in the orientation guess
        # the line above just refused to make.
        full = cv2.resize(pred, (ori_shape[1], ori_shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        lo, hi = float(full.min()), float(full.max())
        normalized = (full - lo) / (hi - lo + 1e-6)

        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
        color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)
        stem = os.path.splitext(image_file_name)[0]
        out_img = os.path.join(save_dir_path, f"{stem}_{model_name}_{precision}_trt.jpg")
        cv2.imwrite(out_img, color_depth_bgr)
        np.savez_compressed(os.path.join(save_dir_path, f"{stem}_{model_name}_trt"),
                            prediction=full)
        print(f"[MDET] saved {os.path.relpath(out_img, CUR_DIR)}")

        common.free_buffers(inputs, outputs, stream)


if __name__ == "__main__":
    main()
