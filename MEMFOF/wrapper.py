# by yhpark 2025-8-10
import sys
import os
import math
import torch
import torch.nn.functional as F

sys.path.insert(1, os.path.join(sys.path[0], "memfof"))
from memfof.core.memfof import *

def _bilinear_sample(feature_map: torch.Tensor, x_coords: torch.Tensor, y_coords: torch.Tensor) -> torch.Tensor:
    """
    feature_map: [B, C, Hf, Wf]
    x_coords: [B*, Hg, Wg] in pixel coordinates (0..Wf-1)
    y_coords: [B*, Hg, Wg] in pixel coordinates (0..Hf-1)
    returns: [B*, C, Hg, Wg]
    """
    b, c, hf, wf = feature_map.shape
    hg, wg = y_coords.shape[-2:]

    x0 = torch.floor(x_coords)
    x1 = x0 + 1.0
    y0 = torch.floor(y_coords)
    y1 = y0 + 1.0

    wx = x_coords - x0
    wy = y_coords - y0
    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy

    valid_x0 = (x0 >= 0) & (x0 <= (wf - 1))
    valid_x1 = (x1 >= 0) & (x1 <= (wf - 1))
    valid_y0 = (y0 >= 0) & (y0 <= (hf - 1))
    valid_y1 = (y1 >= 0) & (y1 <= (hf - 1))

    x0c = x0.clamp(0, wf - 1).long()
    x1c = x1.clamp(0, wf - 1).long()
    y0c = y0.clamp(0, hf - 1).long()
    y1c = y1.clamp(0, hf - 1).long()

    fmap_flat = feature_map.reshape(b, c, hf * wf)

    idx00 = (y0c * wf + x0c).reshape(b, 1, hg * wg)
    idx01 = (y0c * wf + x1c).reshape(b, 1, hg * wg)
    idx10 = (y1c * wf + x0c).reshape(b, 1, hg * wg)
    idx11 = (y1c * wf + x1c).reshape(b, 1, hg * wg)

    v00 = torch.gather(fmap_flat, 2, idx00.expand(b, c, hg * wg)).reshape(b, c, hg, wg)
    v01 = torch.gather(fmap_flat, 2, idx01.expand(b, c, hg * wg)).reshape(b, c, hg, wg)
    v10 = torch.gather(fmap_flat, 2, idx10.expand(b, c, hg * wg)).reshape(b, c, hg, wg)
    v11 = torch.gather(fmap_flat, 2, idx11.expand(b, c, hg * wg)).reshape(b, c, hg, wg)

    w00e = w00.unsqueeze(1)
    w01e = w01.unsqueeze(1)
    w10e = w10.unsqueeze(1)
    w11e = w11.unsqueeze(1)

    m00 = (valid_x0 & valid_y0).to(feature_map.dtype).unsqueeze(1)
    m01 = (valid_x1 & valid_y0).to(feature_map.dtype).unsqueeze(1)
    m10 = (valid_x0 & valid_y1).to(feature_map.dtype).unsqueeze(1)
    m11 = (valid_x1 & valid_y1).to(feature_map.dtype).unsqueeze(1)

    return v00 * w00e * m00 + v01 * w01e * m01 + v10 * w10e * m10 + v11 * w11e * m11

class CorrBlockONNX:
    def __init__(self, fmap1, fmap2, corr_levels, corr_radius):
        self.num_levels = int(corr_levels)
        self.radius = int(corr_radius)
        self.corr_pyramid = []

        fmap2_lvl = fmap2
        for _ in range(self.num_levels):
            corr = CorrBlockONNX.corr(fmap1, fmap2_lvl, num_head=1)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
            self.corr_pyramid.append(corr)
            fmap2_lvl = F.interpolate(
                fmap2_lvl, scale_factor=0.5, mode="bilinear", align_corners=False
            )

    def __call__(self, coords, dilation=None):
        r = self.radius
        # [B, 2, H, W] -> [B, H, W, 2]
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        if dilation is None:
            dilation = torch.ones(
                batch, 1, h1, w1, device=coords.device, dtype=coords.dtype
            )

        dx = torch.arange(-r, r + 1, device=coords.device, dtype=coords.dtype)
        dy = torch.arange(-r, r + 1, device=coords.device, dtype=coords.dtype)
        yy, xx = torch.meshgrid(dy, dx, indexing="ij")
        delta = torch.stack([yy, xx], dim=-1)  # [y, x]

        out_pyramid = []
        for i in range(self.num_levels):
            corr_lvl = self.corr_pyramid[i]

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / float(2 ** i)
            delta_lvl = delta.reshape(1, 2 * r + 1, 2 * r + 1, 2)
            scaled_delta = delta_lvl * dilation.reshape(batch * h1 * w1, 1, 1, 1)
            coords_lvl = centroid_lvl + scaled_delta

            xgrid = coords_lvl[..., 0]
            ygrid = coords_lvl[..., 1]
            sampled = _bilinear_sample(corr_lvl, xgrid, ygrid)
            sampled = sampled.reshape(batch, h1, w1, -1)
            out_pyramid.append(sampled)

        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2, num_head=1):
        batch, dim, h1, w1 = fmap1.shape
        h2, w2 = fmap2.shape[2:]

        fmap1_heads = fmap1.reshape(batch, num_head, dim // num_head, h1 * w1)
        fmap2_heads = fmap2.reshape(batch, num_head, dim // num_head, h2 * w2)

        corr = fmap1_heads.transpose(2, 3) @ fmap2_heads  # [B, H1*W1, H2*W2]
        corr = corr.reshape(batch, num_head, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)

        scale = 1.0 / float(math.sqrt(dim))
        return corr * scale

class MEMFOFModelWrapper(MEMFOF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        images: torch.Tensor,
        iters: int = 8,
        flow_gts: torch.Tensor | None = None,
        fmap_cache: list[torch.Tensor | None] = [None, None, None],
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        """Forward pass of the MEMFOF model.

        Parameters
        ----------
        images : torch.Tensor
            Tensor of shape [B, 3, 3, H, W].
            Images should be in range [0, 255].
        iters : int, optional
            Number of iterations for flow refinement, by default 8
        flow_gts : torch.Tensor | None, optional
            Ground truth flow fields of shape [B, 2, 2, H, W], by default None
            First dimension of size 2 represents backward and forward flows
            Second dimension of size 2 represents x and y components
        fmap_cache : list[torch.Tensor | None], optional
            Cache for feature maps to be used in current forward pass, by default [None, None, None]

        Returns
        -------
        dict[str, torch.Tensor | list[torch.Tensor] | None]
            Dictionary containing:
            - "flow": List of flow predictions of shape [B, 2, 2, H, W] at each iteration
            - "info": List of additional information of shape [B, 2, 4, H, W] at each iteration
            - "nf": List of negative free energy losses of shape [B, 2, 2, H, W] at each iteration (only during training)
            - "fmap_cache": Feature map cache of this forward pass
        """
        B, _, _, H, W = images.shape
        if flow_gts is None:
            flow_gts = torch.zeros(B, 2, 2, H, W, device=images.device)

        images = 2 * (images / 255.0) - 1.0
        images = images.contiguous()

        flow_predictions = []
        info_predictions = []

        # padding
        padder = InputPadder(images.shape)
        images = padder.pad(images)
        B, _, _, H, W = images.shape
        dilation = torch.ones(B, 1, H // 16, W // 16, device=images.device)

        # run the context network
        cnet = self.cnet(torch.cat([images[:, 0], images[:, 1], images[:, 2]], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.dim, self.dim], dim=1)
        attention = self.att(context)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = 0.25 * self.upsample_weight(net)

        flow_16x_21 = flow_update[:, 0:2]
        info_16x_21 = flow_update[:, 2:6]

        flow_16x_23 = flow_update[:, 6:8]
        info_16x_23 = flow_update[:, 8:12]

        if self.training or iters == 0:
            flow_up_21, info_up_21 = self._upsample_data(
                flow_16x_21, info_16x_21, weight_update[:, : 16 * 16 * 9]
            )
            flow_up_23, info_up_23 = self._upsample_data(
                flow_16x_23, info_16x_23, weight_update[:, 16 * 16 * 9 :]
            )
            flow_predictions.append(torch.stack([flow_up_21, flow_up_23], dim=1))
            info_predictions.append(torch.stack([info_up_21, info_up_23], dim=1))

        if iters > 0:
            # run the feature network
            fmap1_16x = (
                self.fnet(images[:, 0])
                if fmap_cache[0] is None
                else fmap_cache[0].clone().to(cnet)
            )
            fmap2_16x = (
                self.fnet(images[:, 1])
                if fmap_cache[1] is None
                else fmap_cache[1].clone().to(cnet)
            )
            fmap3_16x = (
                self.fnet(images[:, 2])
                if fmap_cache[2] is None
                else fmap_cache[2].clone().to(cnet)
            )
            corr_fn_21 = CorrBlockONNX(
                fmap2_16x, fmap1_16x, self.corr_levels, self.corr_radius
            )
            corr_fn_23 = CorrBlockONNX(
                fmap2_16x, fmap3_16x, self.corr_levels, self.corr_radius
            )

        for itr in range(iters):
            B, _, H, W = flow_16x_21.shape
            flow_16x_21 = flow_16x_21.detach()
            flow_16x_23 = flow_16x_23.detach()

            coords21 = (
                coords_grid(B, H, W, device=images.device) + flow_16x_21
            ).detach()
            coords23 = (
                coords_grid(B, H, W, device=images.device) + flow_16x_23
            ).detach()

            corr_21 = corr_fn_21(coords21, dilation=dilation)
            corr_23 = corr_fn_23(coords23, dilation=dilation)

            corr = torch.cat([corr_21, corr_23], dim=1)
            flow_16x = torch.cat([flow_16x_21, flow_16x_23], dim=1)

            net = self.update_block(net, context, corr, flow_16x, attention)

            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)

            flow_16x_21 = flow_16x_21 + flow_update[:, 0:2]
            info_16x_21 = flow_update[:, 2:6]

            flow_16x_23 = flow_16x_23 + flow_update[:, 6:8]
            info_16x_23 = flow_update[:, 8:12]

            if self.training or itr == iters - 1:
                flow_up_21, info_up_21 = self._upsample_data(
                    flow_16x_21, info_16x_21, weight_update[:, : 16 * 16 * 9]
                )
                flow_up_23, info_up_23 = self._upsample_data(
                    flow_16x_23, info_16x_23, weight_update[:, 16 * 16 * 9 :]
                )
                flow_predictions.append(torch.stack([flow_up_21, flow_up_23], dim=1))
                info_predictions.append(torch.stack([info_up_21, info_up_23], dim=1))

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        new_fmap_cache = [None, None, None]
        if iters > 0:
            new_fmap_cache[0] = fmap1_16x.clone().cpu()
            new_fmap_cache[1] = fmap2_16x.clone().cpu()
            new_fmap_cache[2] = fmap3_16x.clone().cpu()

        '''
        if not self.training:
            return {
                "flow": flow_predictions,
                "info": info_predictions,
                "nf": None,
                "fmap_cache": new_fmap_cache,
            }
        '''

        return flow_predictions
