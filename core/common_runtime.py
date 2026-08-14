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

import ctypes
from typing import Optional, List, Union

import numpy as np
import tensorrt as trt

try:
    # cuda-python < 13
    from cuda import cuda, cudart
except ImportError:
    try:
        # Newer layout, when cuda-bindings ships the driver/runtime modules
        from cuda.bindings import driver as cuda, runtime as cudart
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "Could not import the CUDA driver/runtime bindings.\n"
            "cuda-python 13 removed the top-level `from cuda import cuda, cudart`, "
            "and its cuda-bindings wheel does not always ship the driver/runtime "
            "modules either.\n"
            "Install the 12.x line instead:\n"
            '    pip install "cuda-python<13"'
        ) from exc

def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))

        # Wrap the allocation as raw bytes first, then re-interpret. Going
        # straight through np.ctypeslib.as_ctypes_type() fails for float16
        # (no ctypes equivalent); the previous workaround cast the pointer to
        # float32 and built a `size`-element array, which addressed 2x the
        # bytes actually allocated.
        byte_ptr = ctypes.cast(host_mem, ctypes.POINTER(ctypes.c_uint8))
        self._host = np.ctypeslib.as_array(byte_ptr, (nbytes,)).view(dtype)
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: Union[np.ndarray, bytes]):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                )
            np.copyto(self.host[:data.size], data.flat, casting='safe')
        else:
            assert self.host.dtype == np.uint8
            self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


def _resolve_shape_override(binding, shape, size, output_shape):
    """Return the shape to allocate for `binding`, or None to keep the engine's.

    `output_shape` may be:
      * a dict  {binding_name: shape}  — looked up by name
      * a single shape                 — applied only when the engine's own
                                         shape is unusable (dynamic or volume<=1)
    """
    if output_shape is None:
        return None

    if isinstance(output_shape, dict):
        return output_shape.get(binding)

    engine_shape_usable = all(s >= 0 for s in shape) and size > 1
    return None if engine_shape_usable else output_shape


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
def allocate_buffers(engine: trt.ICudaEngine, output_shape:np.ndarray = None, profile_idx: Optional[int] = None):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        shape = engine.get_tensor_shape(binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
        
        size = trt.volume(shape)

        # Some ONNX graphs report a degenerate or partially dynamic shape for an
        # output, so the caller may supply the true shape. Previously this was a
        # list of hardcoded binding names (pts_3d, confidence, points, normal,
        # mask, metric_scale, output) which silently did nothing for any other
        # model. Resolve it by name instead.
        override = _resolve_shape_override(binding, shape, size, output_shape)
        if override is not None:
            size = trt.volume(override)

        trt_type = engine.get_tensor_dtype(binding)

        # Allocate host and device buffers
        if trt.nptype(trt_type):
            dtype = np.dtype(trt.nptype(trt_type))
            bindingMemory = HostDeviceMem(size, dtype)
        else: # no numpy support: create a byte array instead (BF16, FP8, INT4)
            size = int(size * trt_type.itemsize)
            bindingMemory = HostDeviceMem(size)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)
    return inputs, outputs, bindings, stream


# Frees the resources allocated in allocate_buffers
def free_buffers(inputs: List[HostDeviceMem], outputs: List[HostDeviceMem], stream: cudart.cudaStream_t):
    for mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


class StageTimer:
    """CUDA events between the three phases of one inference.

    The loop in core/bench times a whole do_inference and stops after the
    stream synchronizes. That is what a caller waits for, and it is the right
    number to publish -- but it cannot say where the time went, and some
    questions are only about that. Whether a uint8 input is worth it, for
    instance, is entirely a question of whether the smaller host-to-device copy
    pays for the Cast and Transpose it adds to the graph. One total cannot
    answer that; two totals that differ by 3% cannot either.

    Events are recorded on the same stream, between the phases already there,
    so nothing about the work changes. They are created once and reused --
    creating four per iteration would add its own cost to the measurement.

    Note these are GPU-side durations. They will not sum to the wall clock the
    outer loop reports, which also carries launch overhead and the final
    synchronize. That gap is itself worth looking at: on a small model it can
    be most of the time.
    """

    STAGES = ("h2d_ms", "compute_ms", "d2h_ms")

    def __init__(self):
        self._events = [cuda_call(cudart.cudaEventCreate()) for _ in range(4)]
        self.last = {}

    def mark(self, i, stream):
        cuda_call(cudart.cudaEventRecord(self._events[i], stream))

    def read(self):
        """Milliseconds per phase. Only valid after the stream has synchronized."""
        self.last = {
            name: float(cuda_call(cudart.cudaEventElapsedTime(
                self._events[i], self._events[i + 1])))
            for i, name in enumerate(self.STAGES)
        }
        return self.last

    def free(self):
        for e in self._events:
            cuda_call(cudart.cudaEventDestroy(e))
        self._events = []


def _do_inference_base(inputs, outputs, stream, execute_async_func, timer=None):
    if timer is not None:
        timer.mark(0, stream)
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
    if timer is not None:
        timer.mark(1, stream)
    # Run inference.
    execute_async_func()
    if timer is not None:
        timer.mark(2, stream)
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)) for out in outputs]
    if timer is not None:
        timer.mark(3, stream)
    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    if timer is not None:
        timer.read()
    # Return only the host outputs.
    return [out.host for out in outputs]


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, engine, bindings, inputs, outputs, stream, timer=None):
    def execute_async_func():
        context.execute_async_v3(stream_handle=stream)
    # Setup context tensor address.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    return _do_inference_base(inputs, outputs, stream, execute_async_func, timer)