#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import hashlib
import json
import os
import time

import tensorrt as trt
from common_runtime import *

import sys

# Appended, never inserted at 0. Model scripts add the repo root at index 1 on
# purpose so that sys.path[0] stays their own directory, and at least one of
# them reads sys.path[0] afterwards to reach a vendored package:
# metric_anything does
#     sys.path.insert(1, os.path.join(sys.path[0], "metric_anything/models/..."))
# to find its copy of `moge`. Inserting the root at 0 here shifted that out
# from under it and the script died on ModuleNotFoundError -- from a file it
# does not import directly, several imports earlier. Appending cannot shadow a
# module or move an existing entry, and `core` only exists at the root anyway.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
from core import build_conditions

try:
    # Sometimes python does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def GiB(val):
    return val * 1 << 30






# Sets up the builder to use the timing cache file, and creates it if it does not already exist
def setup_timing_cache(config: trt.IBuilderConfig, timing_cache_path: os.PathLike):
    buffer = b""
    if os.path.exists(timing_cache_path):
        with open(timing_cache_path, mode="rb") as timing_cache_file:
            buffer = timing_cache_file.read()
    timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
    config.set_timing_cache(timing_cache, True)


# Saves the config's timing cache to file
def save_timing_cache(config: trt.IBuilderConfig, timing_cache_path: os.PathLike):
    timing_cache: trt.ITimingCache = config.get_timing_cache()
    with open(timing_cache_path, "wb") as timing_cache_file:
        timing_cache_file.write(memoryview(timing_cache.serialize()))


# ---------------------------------------------------------------------------
# Engine build / load
#
# Replaces the 15 near-duplicate get_engine() copies that lived in each model's
# onnx2trt.py. They shared a signature and feature set, but three differences
# were real and are now parameters:
#   * workspace     2 GiB default, 3 GiB for Depth_Pro pointcloud, 4 GiB for VGGT/StreamVGGT
#   * OBEY_PRECISION_CONSTRAINTS   metric3d_v2 only
#   * a `plan is None` guard       MoGe_2 only (now always on)
# ---------------------------------------------------------------------------

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def _engine_fingerprint(onnx_file_path, precision, workspace_gib, opt_level,
                        obey_precision_constraints, dynamic_input_shapes):
    """Identify what an engine was built from, so a stale one is not reused."""
    import hashlib

    h = hashlib.sha256()
    with open(onnx_file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    parts = [
        h.hexdigest(),
        f"trt={trt.__version__}",
        f"precision={precision}",
        f"workspace={workspace_gib}",
        f"opt_level={opt_level}",
        f"obey_precision={obey_precision_constraints}",
        f"dynamic={dynamic_input_shapes}",
    ]
    try:
        import torch

        if torch.cuda.is_available():
            parts.append(f"gpu={torch.cuda.get_device_name(0)}")
    except Exception:
        pass
    return "\n".join(parts)


def engine_staleness(engine_file_path, fingerprint_path, fingerprint, onnx_exists):
    """Why the cached engine cannot be reused, or None if it can.

    Split out from get_engine() so the decision is testable without triggering
    a multi-minute rebuild.
    """
    if not os.path.exists(engine_file_path):
        return "no engine file"
    if not onnx_exists:
        # Engine-only deployment: nothing to validate against, so trust it.
        return None
    if fingerprint is None:
        return None
    if not os.path.exists(fingerprint_path):
        return "no fingerprint recorded"
    with open(fingerprint_path, encoding="utf-8") as f:
        if f.read() != fingerprint:
            return "ONNX or build options changed"
    return None


def get_engine(
    onnx_file_path,
    engine_file_path="",
    precision="fp32",
    dynamic_input_shapes=None,
    workspace_gib=2,
    opt_level=None,
    obey_precision_constraints=False,
    check_fingerprint=True,
):
    """Load `engine_file_path` if it matches, otherwise build it from ONNX.

    `opt_level` is TensorRT's builder_optimization_level (0-5, default 3).
    Lower it when a build stalls or runs out of memory on a small GPU; raise it
    to trade build time for inference speed. Pass it explicitly — never adjust
    it automatically, or benchmark numbers stop being comparable.
    """
    fingerprint = None
    fingerprint_path = os.path.splitext(engine_file_path)[0] + ".fingerprint"
    if check_fingerprint and os.path.exists(onnx_file_path):
        fingerprint = _engine_fingerprint(
            onnx_file_path, precision, workspace_gib, opt_level,
            obey_precision_constraints, dynamic_input_shapes,
        )

    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(0) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser, \
                trt.Runtime(TRT_LOGGER) as runtime:

            if not os.path.exists(onnx_file_path):
                raise FileNotFoundError(f"[MDET] ONNX file {onnx_file_path} not found.")

            # The old copies ignored parse_from_file()'s return value, so a
            # malformed ONNX produced an empty network and a confusing failure.
            if not parser.parse_from_file(onnx_file_path):
                errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"[MDET] Failed to parse {onnx_file_path}:\n{errors}")

            # The cache name carries the fingerprint, which carries the ONNX
            # hash and the builder options. A timing cache is measured tactic
            # timings, and TensorRT trusts them instead of re-timing -- so a
            # cache left over from a different graph can make it pick a kernel
            # that is no longer the fast one, and nothing detects that.
            #
            # This is not hypothetical. distill_any_depth shipped an engine
            # that ran 86.6% of its time in fp32 layers and took 10.02 ms;
            # deleting the engine and its cache and rebuilding from the same
            # ONNX with the same options produced a 1.5%-fp32 engine at
            # 4.31 ms. The fingerprint matched throughout, so the engine was
            # being reused, and the cache was the only input not covered by it.
            tag = hashlib.sha256((fingerprint or engine_file_path).encode()).hexdigest()[:12]
            timing_cache = os.path.join(
                os.path.dirname(engine_file_path),
                f"{os.path.splitext(os.path.basename(engine_file_path))[0]}_{tag}_timing.cache",
            )
            setup_timing_cache(config, timing_cache)

            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(workspace_gib))

            # SPARSE_WEIGHTS was set on most models but does nothing unless the
            # weights were trained with 2:4 structured sparsity, which none of
            # these are. Dropped rather than kept as a no-op.

            if precision == "fp16":
                # platform_has_fast_fp16 was deprecated in TensorRT 10 and
                # removed in 11, where reading it raises AttributeError. It was
                # only ever advisory -- setting the flag on a GPU without fast
                # fp16 is allowed, TensorRT just may not use it. Default to
                # True when the attribute is gone.
                if getattr(builder, "platform_has_fast_fp16", True):
                    config.set_flag(trt.BuilderFlag.FP16)
                    print("[MDET] set fp16 model")
                else:
                    print("[MDET] platform has no fast fp16; building fp32")
            if obey_precision_constraints:
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                print("[MDET] set OBEY_PRECISION_CONSTRAINTS")
            if opt_level is not None:
                config.builder_optimization_level = opt_level
                print(f"[MDET] builder optimization level {opt_level}")

            if dynamic_input_shapes is not None:
                profile = builder.create_optimization_profile()
                for i_idx in range(network.num_inputs):
                    inp = network.get_input(i_idx)
                    assert inp.shape[0] == -1
                    min_shape, opt_shape, max_shape = dynamic_input_shapes[:3]
                    profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
                    print(f"[MDET] Input '{inp.name}' Optimization Profile with shape "
                          f"MIN {min_shape} / OPT {opt_shape} / MAX {max_shape}")
                config.add_optimization_profile(profile)

            for i_idx in range(network.num_inputs):
                t = network.get_input(i_idx)
                print(f"[MDET] input({i_idx}) name: {t.name}, shape= {t.shape}")
            for o_idx in range(network.num_outputs):
                t = network.get_output(o_idx)
                print(f"[MDET] output({o_idx}) name: {t.name}, shape= {t.shape}")

            clock_before, clock_max = build_conditions.gpu_clock()
            plan = builder.build_serialized_network(network, config)
            clock_after, _ = build_conditions.gpu_clock()
            if plan is None:
                raise RuntimeError(
                    "[MDET] build_serialized_network returned None — see the TensorRT log "
                    "above. On a small GPU try a lower workspace_gib or opt_level."
                )
            engine = runtime.deserialize_cuda_engine(plan)
            save_timing_cache(config, timing_cache)
            os.makedirs(os.path.dirname(engine_file_path) or ".", exist_ok=True)
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            if fingerprint is not None:
                with open(fingerprint_path, "w", encoding="utf-8") as f:
                    f.write(fingerprint)
            info = {
                "clock_before_mhz": clock_before,
                "clock_after_mhz": clock_after,
                "clock_max_mhz": clock_max,
                "tensorrt": trt.__version__,
            }
            with open(build_conditions.path_for(engine_file_path), "w",
                      encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            complaint = build_conditions.complaint(info)
            if complaint:
                print(f"[MDET] WARNING: this engine was {complaint}. "
                      f"Pin the clock (nvidia-smi -lgc 1800,1800) and rebuild "
                      f"before treating its speed as this model's speed.")
            return engine

    stale = engine_staleness(
        engine_file_path, fingerprint_path, fingerprint, os.path.exists(onnx_file_path)
    )
    if os.path.exists(engine_file_path):
        if stale is None:
            print(f"[MDET] Load engine from file ({engine_file_path})")
            info = None
            info_path = build_conditions.path_for(engine_file_path)
            if os.path.exists(info_path):
                try:
                    with open(info_path, encoding="utf-8") as f:
                        info = json.load(f)
                except (OSError, ValueError):
                    info = None
            complaint = build_conditions.complaint(
                info, now_mhz=build_conditions.gpu_clock()[0])
            if complaint:
                # Reused rather than rebuilt: an engine built under a loose
                # clock still computes the right answer, and forcing a
                # multi-minute rebuild on every run of every script would get
                # this check switched off. Saying so is the job here.
                print(f"[MDET] WARNING: engine {complaint}.")
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        # The old code loaded any engine that merely existed, so edits to the
        # ONNX or the builder options were silently ignored.
        # ASCII: this prints on a cp949 console, where an em dash raises inside
        # print and replaces the explanation with a UnicodeEncodeError.
        print(f"[MDET] Rebuilding engine - {stale}")

    print(f"[MDET] Build engine ({engine_file_path})")
    begin = time.time()
    engine = build_engine()
    dur = time.time() - begin
    dur_str = f"{dur:.2f} [sec]" if dur < 60 else f"{dur // 60:.0f} [min] {dur % 60:.2f} [sec]"
    print(f"[MDET] Engine build done! ({dur_str})")
    return engine
    