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

import torch


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

    Pass the PositionGetter classes to patch; VGGT and StreamVGGT each vendor
    their own copy of rope.py.
    """
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
