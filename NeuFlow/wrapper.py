# by yhpark 2025-8-9
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(1, os.path.join(sys.path[0], "NeuFlow_v2"))
from NeuFlow_v2.NeuFlow.neuflow import NeuFlow
from NeuFlow_v2.NeuFlow import utils

# Avoids grid_sample by using flattened gather-based bilinear sampling
def _bilinear_sample_flat(img: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bilinear sample without grid_sample.
    img: [N, C, H, W]
    x:   [N, Hg, Wg] in pixel coords (0..W-1)
    y:   [N, Hg, Wg] in pixel coords (0..H-1)
    return: [N, C, Hg, Wg]
    """
    n, c, h, w = img.shape
    hg, wg = y.shape[-2:]

    x0 = torch.floor(x)
    x1 = x0 + 1.0
    y0 = torch.floor(y)
    y1 = y0 + 1.0

    wx = x - x0
    wy = y - y0
    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy

    valid_x0 = (x0 >= 0) & (x0 <= (w - 1))
    valid_x1 = (x1 >= 0) & (x1 <= (w - 1))
    valid_y0 = (y0 >= 0) & (y0 <= (h - 1))
    valid_y1 = (y1 >= 0) & (y1 <= (h - 1))

    x0c = x0.clamp(0, w - 1).long()
    x1c = x1.clamp(0, w - 1).long()
    y0c = y0.clamp(0, h - 1).long()
    y1c = y1.clamp(0, h - 1).long()

    img_flat = img.view(n, c, h * w)

    idx00 = (y0c * w + x0c).view(n, 1, hg * wg)
    idx01 = (y0c * w + x1c).view(n, 1, hg * wg)
    idx10 = (y1c * w + x0c).view(n, 1, hg * wg)
    idx11 = (y1c * w + x1c).view(n, 1, hg * wg)

    v00 = torch.gather(img_flat, 2, idx00.expand(n, c, hg * wg)).view(n, c, hg, wg)
    v01 = torch.gather(img_flat, 2, idx01.expand(n, c, hg * wg)).view(n, c, hg, wg)
    v10 = torch.gather(img_flat, 2, idx10.expand(n, c, hg * wg)).view(n, c, hg, wg)
    v11 = torch.gather(img_flat, 2, idx11.expand(n, c, hg * wg)).view(n, c, hg, wg)

    w00e = w00.unsqueeze(1)
    w01e = w01.unsqueeze(1)
    w10e = w10.unsqueeze(1)
    w11e = w11.unsqueeze(1)

    m00 = (valid_x0 & valid_y0).to(img.dtype).unsqueeze(1)
    m01 = (valid_x1 & valid_y0).to(img.dtype).unsqueeze(1)
    m10 = (valid_x0 & valid_y1).to(img.dtype).unsqueeze(1)
    m11 = (valid_x1 & valid_y1).to(img.dtype).unsqueeze(1)

    return v00 * w00e * m00 + v01 * w01e * m01 + v10 * w10e * m10 + v11 * w11e * m11


class CorrBlockONNX:
    def __init__(self, radius, levels):
        self.radius = int(radius)
        self.levels = int(levels)

    def init_bhwd(self, batch_size, height, width, device, amp):
        xy_range = torch.linspace(-self.radius, self.radius, 2 * self.radius + 1,
                                  dtype=torch.half if amp else torch.float,
                                  device=device)
        # delta: [1, K, K, 2] with indexing='ij'
        delta = torch.stack(torch.meshgrid(xy_range, xy_range, indexing='ij'), axis=-1)
        delta = delta.view(1, 2 * self.radius + 1, 2 * self.radius + 1, 2)

        self.grid = utils.coords_grid(batch_size, height, width, device, amp)
        self.delta = delta.repeat(batch_size * height * width, 1, 1, 1)

    def __call__(self, corr_pyramid, flow):
        b, _, h, w = flow.shape
        coords = (self.grid + flow).permute(0, 2, 3, 1)
        coords = coords.reshape(b * h * w, 1, 1, 2)

        out_list = []
        for level, corr in enumerate(corr_pyramid):
            curr_coords = coords / (2 ** level) + self.delta
            xgrid = curr_coords[..., 0]
            ygrid = curr_coords[..., 1]
            sampled = _bilinear_sample_flat(corr, xgrid, ygrid)
            sampled = sampled.view(b, h, w, -1)
            out_list.append(sampled)

        out = torch.cat(out_list, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous()

    def init_corr_pyr(self, feature0: torch.Tensor, feature1: torch.Tensor):
        b, c, h, w = feature0.shape
        feature0 = feature0.view(b, c, h * w)
        feature1 = feature1.view(b, c, h * w)

        corr = torch.matmul(feature0.transpose(1, 2), feature1)
        corr = corr.view(b * h * w, 1, h, w) / torch.sqrt(
            torch.tensor(c, dtype=corr.dtype, device=corr.device)
        )

        corr_pyramid = [corr]
        for _ in range(self.levels - 1):
            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
            corr_pyramid.append(corr)

        return corr_pyramid


class NeuFlowModelWrapper(NeuFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corr_block_s16 = CorrBlockONNX(radius=4, levels=1)
        self.corr_block_s8 = CorrBlockONNX(radius=4, levels=1)