# by yhpark 2026-8-14
# TR2M TensorRT model generation
"""Build the TR2M engine, benchmark it, and write a depth image.

The one model here with two inputs. The image goes in as usual; the second
input is a [1, 1, 768] CLIP embedding of a sentence describing the scene,
which is what TR2M uses to decide the metric scale. CLIP itself is not in the
graph, so this script never loads it -- onnx_export.py wrote the embedding to
onnx/<name>_text_features.npy and this reads it back. That is why this stage
runs in the shared TensorRT environment like every other model's.

The prompt is therefore fixed for the benchmark, and deliberately so: a
measurement whose input depends on what someone typed is not a measurement.
Changing it means re-running onnx_export.py, which records the new sentence
next to the new embedding.

Note 434x560, not the 518x518 most of the others use. Frames per second
between the two is a comparison between different amounts of work.
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

import common  # noqa: E402
from common import *  # noqa: E402,F401,F403
from core import bench  # noqa: E402

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(raw_bgr, input_h, input_w):
    """BGR uint8 -> normalised float32 NCHW, stretched to the network's size.

    A plain resize, matching what upstream does for stills. It distorts a
    16:9 photo, and the alternative -- centre-cropping to 560/434 -- throws
    away the sides. Upstream chose the distortion for still images and the
    benchmark follows it rather than inventing a third answer.
    """
    img = cv2.resize(raw_bgr, (input_w, input_h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


def main():
    iteration = 100
    warmup = 20
    precision = "fp16"              # 'fp32' or 'fp16'
    input_h, input_w = 434, 560
    depth_model = "da_s"
    img_encoder = "vitl"
    encoder = "vits"                # the Depth Anything backbone, for the record
    min_depth, max_depth = 0.3, 10.0    # metres, for the colour map only

    model_name = f"tr2m_{depth_model}_{img_encoder}_{input_h}x{input_w}"
    onnx_model_path = os.path.join(CUR_DIR, "onnx", f"{model_name}.onnx")
    text_path = os.path.join(CUR_DIR, "onnx", f"{model_name}_text_features.npy")
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
    batch_images = preprocess_image(raw_image, input_h, input_w)
    print(f"[MDET] after preprocess shape : {batch_images.shape}")

    if not os.path.exists(text_path):
        raise FileNotFoundError(
            f"[MDET] {text_path} not found. Run onnx_export.py first -- it writes "
            f"the prompt embedding beside the ONNX so this stage needs no CLIP.")
    text_features = np.ascontiguousarray(np.load(text_path).astype(np.float32))
    print(f"[MDET] text features : {text_features.shape}")

    with get_engine(onnx_model_path, engine_file_path, precision) as engine, \
            engine.create_execution_context() as context:

        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Order follows the ONNX input names: image first, text second.
        inputs[0].host = batch_images
        inputs[1].host = text_features

        def run():
            return common.do_inference(context, engine=engine, bindings=bindings,
                                       inputs=inputs, outputs=outputs, stream=stream)

        trt_outputs, samples = bench.measure(run, warmup=warmup, iterations=iteration)

        # metric_depth, relative_depth, scale_map, shift_map -- see spec.json
        metric = np.array(trt_outputs[0]).reshape(input_h, input_w)
        relative = np.array(trt_outputs[1]).reshape(input_h, input_w)

        bench.record("tr2m", samples, encoder=encoder, warmup=warmup,
                     precision=precision, profile="bench",
                     input_h=input_h, input_w=input_w,
                     engine_path=engine_file_path,
                     outputs={"metric_depth": metric, "relative_depth": relative},
                     model_input=batch_images)
        print(f"[MDET] metric depth : min {metric.min():.3f} m, "
              f"max {metric.max():.3f} m, mean {metric.mean():.3f} m")

        # ===================================================================
        print("[MDET] Generate color depth image")
        depth_full = cv2.resize(metric, (ori_shape[1], ori_shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        # Near is the warm end. Inverse depth rather than depth, so the near
        # field -- where a metric model's error actually matters -- is not
        # compressed into one colour by a distant wall.
        inverse = 1.0 / np.clip(depth_full, min_depth, max_depth)
        lo, hi = 1.0 / max_depth, 1.0 / min_depth
        normalized = (inverse - lo) / (hi - lo + 1e-6)

        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
        color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)
        stem = os.path.splitext(image_file_name)[0]
        out_img = os.path.join(save_dir_path, f"{stem}_{model_name}_{precision}_trt.jpg")
        cv2.imwrite(out_img, color_depth_bgr)
        np.savez_compressed(os.path.join(save_dir_path, f"{stem}_{model_name}_trt"),
                            metric_depth=depth_full, relative_depth=relative)
        print(f"[MDET] saved {os.path.relpath(out_img, CUR_DIR)}")

        common.free_buffers(inputs, outputs, stream)


if __name__ == "__main__":
    main()
