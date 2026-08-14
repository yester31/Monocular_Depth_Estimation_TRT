"""Patches applied only while exporting to ONNX.

Each one exists because a model does something an ONNX exporter cannot
represent. Each is verified against the unpatched PyTorch output before being
trusted: an export that runs but computes something else is worse than one
that fails.

These replace hand-edits. VGGT/onnx_export.py and StreamVGGT/onnx_export.py
used to carry a `### NOTICE ###` telling the reader to go and edit a specific
line of the upstream clone before exporting. That works, but it is invisible
to anyone who runs the script without reading the header, it is silently
undone by `git pull` in the clone, and nothing checks that it was done.
"""

import contextlib
import sys

import torch


def _loaded_position_getters():
    """Every distinct PositionGetter class currently in sys.modules.

    Deduplicated by identity, not by name: the whole point is that the same
    source file can be loaded twice under different module names and produce
    two classes that are not each other.
    """
    found, seen = [], set()
    for name, mod in list(sys.modules.items()):
        if not name.endswith(".rope") or mod is None:
            continue
        cls = getattr(mod, "PositionGetter", None)
        if isinstance(cls, type) and id(cls) not in seen:
            seen.add(id(cls))
            found.append(cls)
    return found


@contextlib.contextmanager
def no_cartesian_prod(*classes):
    """Build RoPE positions without torch.cartesian_prod.

    `PositionGetter.__call__` builds the patch-grid coordinates with

        positions = torch.cartesian_prod(y_coords, x_coords)

    and the TorchScript exporter has no symbolic for it at any opset:

        UnsupportedOperatorError: Exporting the operator
        'aten::cartesian_prod' to ONNX opset version 17 is not supported

    This is the replacement the NOTICE in both onnx_export.py files described,
    applied automatically instead of by editing the upstream clone:

        yy = y_coords.unsqueeze(1).expand(-1, x_coords.size(0))
        xx = x_coords.unsqueeze(0).expand(y_coords.size(0), -1)
        positions = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)

    cartesian_prod of two 1-D tensors is exactly the row-major pairing of every
    y with every x, which is what the expand-and-stack produces, so the values
    are identical -- verified on the real model, see probe_rope.py.

    Pass nothing and every loaded PositionGetter is patched. That default
    matters here. VGGT's clone root and its inner package are **both**
    namespace portions, so `vggt.layers.rope` and `vggt.vggt.layers.rope` are
    two distinct module objects holding two distinct classes -- loaded from
    the same file:

        same module object : False
        same class object  : False
        outer file: ...\\vggt\\vggt\\layers\\rope.py
        inner file: ...\\vggt\\vggt\\layers\\rope.py

    The export script reaches the class by one spelling and the model by the
    other, so patching the one you imported silently does nothing and the
    export fails exactly as if no patch were applied.
    """
    if not classes:
        classes = _loaded_position_getters()
        if not classes:
            raise RuntimeError(
                "no PositionGetter found in sys.modules — import the model "
                "before entering no_cartesian_prod()")
    def patched_call(self, batch_size, height, width, device):
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            yy = y_coords.unsqueeze(1).expand(-1, x_coords.size(0))
            xx = x_coords.unsqueeze(0).expand(y_coords.size(0), -1)
            self.position_cache[height, width] = torch.stack(
                [yy.reshape(-1), xx.reshape(-1)], dim=1)
        cached = self.position_cache[height, width]
        return cached.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

    originals = [(c, c.__call__) for c in classes]
    for c, _ in originals:
        c.__call__ = patched_call
    try:
        yield
    finally:
        for c, real in originals:
            c.__call__ = real


@contextlib.contextmanager
def float32_sincos_pos_embed(*modules):
    """Build VGGT's positional-embedding frequencies in float32, not double.

    `vggt/heads/utils.py:make_sincos_pos_embed` does this:

        omega = torch.arange(embed_dim // 2,
                             dtype=torch.float32 if device.type == "mps"
                                   else torch.double, device=device)
        ...
        out = torch.einsum("m,d->md", pos, omega)     # pos is float32

    Upstream picks float64 on purpose, for precision in `1 / omega_0 ** omega`,
    and casts back with `.float()` at the end. PyTorch handles the mixed
    einsum by type promotion. ONNX cannot: Einsum's type parameter T has to
    bind to one type, so the exporter emits a graph that onnxruntime refuses
    to load at all --

        Type Error: Type parameter (T) of Optype (Einsum) bound to different
        types (tensor(float) and tensor(double)) in node (/depth_head/Einsum_9)

    -- while TensorRT accepts it and, having no fp64, computes in fp32 anyway.

    So the engine was already doing this. Making it explicit costs nothing at
    runtime and produces a graph other tools can read, which is what made the
    accuracy check possible in the first place.

    Pass the modules holding the function; VGGT and StreamVGGT each vendor a
    copy, and the duplicate-namespace problem in no_cartesian_prod applies
    here too, so pass nothing to patch every loaded copy.
    """
    if not modules:
        modules = [m for name, m in list(sys.modules.items())
                   if name.endswith("heads.utils") and m is not None
                   and hasattr(m, "make_sincos_pos_embed")]
        if not modules:
            raise RuntimeError(
                "no make_sincos_pos_embed found in sys.modules — import the "
                "model before entering float32_sincos_pos_embed()")

    def patched(embed_dim, pos, omega_0=100):
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
        omega /= embed_dim / 2.0
        omega = 1.0 / omega_0 ** omega
        pos = pos.reshape(-1).to(torch.float32)
        out = torch.einsum("m,d->md", pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()

    originals = [(m, m.make_sincos_pos_embed) for m in modules]
    for m, _ in originals:
        m.make_sincos_pos_embed = patched
    try:
        yield
    finally:
        for m, real in originals:
            m.make_sincos_pos_embed = real


@contextlib.contextmanager
def no_mono_sky_postprocess(net_class):
    """Drop Depth Anything V3's sky pass from the exported graph.

    `DepthAnything3Net._process_mono_sky_estimation` runs after the network
    and rewrites depth in sky regions:

        non_sky_depth = output.depth[non_sky_mask]     # size depends on data
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), ...)
            sampled_depth = non_sky_depth[idx]
        non_sky_max = torch.quantile(sampled_depth, 0.99)
        ... set sky regions to non_sky_max

    None of that can live in a static graph. The TorchScript exporter says so
    directly:

        SymbolicValueError: ONNX symbolic expected a constant value of the
        'high' argument, got ... onnx::ReduceProd

    because `high` is `numel()` of a boolean-masked tensor. Beyond the export,
    the step draws 100000 random indices, so it is **not deterministic** --
    two PyTorch runs on the same image disagree in the sky by however much the
    sampled 99th percentile moves.

    This is post-process, not the network. The export wrapper already returns
    `depth` and `sky` as separate outputs, so a caller that wants the sky fill
    can do it on the host with the full depth map and no sampling at all.

    **The output does change on images that contain sky**: those pixels keep
    their predicted depth instead of being flattened to the non-sky maximum.

    Measured on data/example.jpg and the difference was 0.000000 across every
    pixel -- but that does not validate the patch. That photo has no sky, so
    the pass returns early at

        if (~non_sky_mask).sum() <= 10: return output

    and never runs. The measurement shows the benchmark image is unaffected,
    nothing more. An outdoor photo would differ, and there is no reference to
    compare against there either, because the step samples 100000 random
    indices and does not reproduce itself between runs.
    """
    real = net_class._process_mono_sky_estimation
    net_class._process_mono_sky_estimation = lambda self, output: output
    try:
        yield
    finally:
        net_class._process_mono_sky_estimation = real


@contextlib.contextmanager
def static_rope_max_position(max_position: int, *classes):
    """Give VGGT-family RoPE a constant table size instead of reading one.

    Only needed on the **dynamo** export path, which this repository does not
    use for these models -- kept because it is verified and because the dynamo
    path is where a future TensorRT-11 port would land.

    `RotaryPositionEmbedding2D.forward` sizes its frequency table like this:

        max_position = int(positions.max()) + 1

    `positions.max()` is a tensor, so `int()` is `.item()`, and torch.export
    turns that into an unbacked symbol it then refuses to specialise:

        create_unbacked_symint u0 ... in local_scalar_dense
        GuardOnDataDependentSymNode: Could not extract specialized integer
        from data-dependent expression u0

    The value is not data-dependent. `positions` is the patch grid, so the
    maximum is always `max(height, width) - 1`, fixed once the input
    resolution is -- and these engines pin the resolution. Verified
    bit-identical on the real model (probe_rope.py): max abs diff 0.0000000000.
    """
    if max_position < 1:
        raise ValueError(f"max_position must be >= 1, got {max_position}")

    def patched_forward(self, tokens, positions):
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, \
            "Positions must have shape (batch_size, n_tokens, 2)"
        feature_dim = tokens.size(-1) // 2
        # the one changed line
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype)
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    originals = [(c, c.forward) for c in classes]
    for c, _ in originals:
        c.forward = patched_forward
    try:
        yield
    finally:
        for c, real in originals:
            c.forward = real


def rope_max_position(input_h: int, input_w: int, patch_size: int = 14) -> int:
    """The constant `int(positions.max()) + 1` would have produced.

    The patch grid is input // patch_size in each direction, positions run
    0..grid-1, and the table is indexed by the larger of the two.
    """
    grid_h, grid_w = input_h // patch_size, input_w // patch_size
    if grid_h < 1 or grid_w < 1:
        raise ValueError(
            f"{input_h}x{input_w} is smaller than one {patch_size}px patch")
    return max(grid_h, grid_w)
