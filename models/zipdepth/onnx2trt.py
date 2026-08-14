# by yhpark 2026-8-14
# ZipDepth TensorRT model generation
"""Build the ZipDepth engine, benchmark it, and write a depth image.

Watch the preprocessing: **this model is /255 and nothing else.** Every other
model here subtracts the ImageNet mean and divides by its standard deviation,
and doing that to ZipDepth would feed it an image it has never seen. The
normalisation lives with the model, not with the repository.

The output is affine-invariant *inverse* depth, so a large value is near. It
carries no unit and no zero, which is why the comparison table lists it as
relative and the ground-truth evaluation fits a scale and a shift before
scoring it.
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
from core import preprocess as pp  # noqa: E402

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


# preprocess_image() is now core/preprocess.py. Upstream's predictor does BGR to
# RGB, divide by 255, and stop -- no mean, no standard deviation. Applying the
# ImageNet statistics here would not crash; it would score badly and look like a
# weak model, which is what tests/test_preprocess_spec.py guards against. The
# INTER_AREA stretch matches upstream's rule (short side to 384, both sides a
# multiple of 32) with the arithmetic already done for a 4:3 source.
# tests/test_preprocess.py asserts np.array_equal against
# reports/inputs/zipdepth.npy.


def main():
    iteration = 100
    warmup = 20
    precision = "fp16"              # 'fp32' or 'fp16'
    input_h, input_w = 384, 512
    variant = "base"
    npu = False

    model_name = f"zipdepth_{variant}{'_npu' if npu else ''}_{input_h}x{input_w}"
    onnx_model_path = os.path.join(CUR_DIR, "onnx", f"{model_name}.onnx")
    engine_file_path = os.path.join(CUR_DIR, "engine", f"{model_name}_{precision}.engine")
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)

    save_dir_path = os.path.join(CUR_DIR, "results")
    os.makedirs(save_dir_path, exist_ok=True)

    # Input
    image_file_name = "example.jpg"
    image_path = os.path.join(_R, "data", image_file_name)
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise FileNotFoundError(f"[MDET] could not read {image_path}")
    ori_shape = raw_image.shape[:2]
    print(f"[MDET] original image size : {ori_shape}")
    batch_images, geom = pp.preprocess_for(raw_image, 'zipdepth',
                                           (input_h, input_w))
    print(f"[MDET] after preprocess shape : {batch_images.shape}")

    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:

        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = batch_images

        def run():
            return common.do_inference(context, engine=engine, bindings=bindings,
                                       inputs=inputs, outputs=outputs, stream=stream)

        trt_outputs, samples = bench.measure(run, warmup=warmup, iterations=iteration)
        inv_depth = np.array(trt_outputs[0]).reshape(input_h, input_w)

        bench.record("zipdepth", samples, encoder=variant, warmup=warmup,
                     precision=precision, profile="bench",
                     input_h=input_h, input_w=input_w,
                     engine_path=engine_file_path,
                     outputs={"inverse_depth": inv_depth},
                     model_input=batch_images)
        print(f"[MDET] inverse depth : min {inv_depth.min():.5f} , "
              f"max {inv_depth.max():.5f}")

        # ===================================================================
        print("[MDET] Generate color depth image")
        # The output is already inverse depth, so it goes straight into the
        # colour map -- no 1/d here. Normalised by its own range because it has
        # no unit: an absolute scale would be inventing one.
        full = cv2.resize(inv_depth, (ori_shape[1], ori_shape[0]),
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
                            inverse_depth=full)
        print(f"[MDET] saved {os.path.relpath(out_img, CUR_DIR)}")

        common.free_buffers(inputs, outputs, stream)


if __name__ == "__main__":
    main()
