"""Patches applied only while exporting to ONNX.

Each one exists because a model does something torch.export or the TorchScript
exporter cannot represent, and each is measured against the unpatched PyTorch
output before being trusted -- an export that runs but computes something else
is worse than one that fails.
"""

import contextlib

import torch


@contextlib.contextmanager
def static_rope_max_position(max_position: int, *classes):
    """Give VGGT-family RoPE a constant table size instead of reading one.

    `RotaryPositionEmbedding2D.forward` sizes its frequency table like this:

        max_position = int(positions.max()) + 1

    `positions.max()` is a tensor, so `int()` is `.item()`, and under
    torch.export that produces an unbacked symbol:

        create_unbacked_symint u0 ... in local_scalar_dense
        GuardOnDataDependentSymNode: Could not extract specialized integer
        from data-dependent expression u0

    The value is not actually data-dependent. `positions` comes from
    `PositionGetter`, which is `cartesian_prod(arange(height), arange(width))`
    over the patch grid, so the maximum is always `max(height, width) - 1` and
    is fixed the moment the input resolution is. Only the trip through a tensor
    makes it opaque, and the fixed-size engines here pin the resolution anyway.

    Pass the classes to patch, since VGGT and StreamVGGT each vendor their own
    copy of this file. The bodies are identical; this reproduces it with the
    one line replaced.

    The other exporter is no better here: TorchScript has no symbolic for
    `aten::cartesian_prod`, which the same file uses to build `positions`. So
    neither path works untouched, and this patch is what makes the dynamo path
    viable.
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
