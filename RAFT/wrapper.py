# by yhpark 2025-8-9
import sys
import os
import math
import torch
import torch.nn.functional as F

sys.path.insert(1, os.path.join(sys.path[0], "RAFT", "core"))
from RAFT.core.raft import *

def _bilinear_sample(feature_map: torch.Tensor, x_coords: torch.Tensor, y_coords: torch.Tensor) -> torch.Tensor:
    """Bilinear sample feature_map at positions (x_coords, y_coords) using flattened gather.
    feature_map: [B, C, Hf, Wf]
    x_coords: [B, Hg, Wg] in pixel coordinates (0..Wf-1)
    y_coords: [B, Hg, Wg] in pixel coordinates (0..Hf-1)
    returns: [B, C, Hg, Wg]
    """
    b, c, hf, wf = feature_map.shape
    hg, wg = y_coords.shape[-2:]

    # neighbors
    x0 = torch.floor(x_coords)
    x1 = x0 + 1.0
    y0 = torch.floor(y_coords)
    y1 = y0 + 1.0

    # weights
    wx = x_coords - x0
    wy = y_coords - y0
    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy

    # validity masks
    valid_x0 = (x0 >= 0) & (x0 <= (wf - 1))
    valid_x1 = (x1 >= 0) & (x1 <= (wf - 1))
    valid_y0 = (y0 >= 0) & (y0 <= (hf - 1))
    valid_y1 = (y1 >= 0) & (y1 <= (hf - 1))

    x0c = x0.clamp(0, wf - 1).long()
    x1c = x1.clamp(0, wf - 1).long()
    y0c = y0.clamp(0, hf - 1).long()
    y1c = y1.clamp(0, hf - 1).long()

    # flatten feature map and build linear indices for the 4 neighbors
    fmap_flat = feature_map.view(b, c, hf * wf)

    idx00 = (y0c * wf + x0c).view(b, 1, hg * wg)
    idx01 = (y0c * wf + x1c).view(b, 1, hg * wg)
    idx10 = (y1c * wf + x0c).view(b, 1, hg * wg)
    idx11 = (y1c * wf + x1c).view(b, 1, hg * wg)

    v00 = torch.gather(fmap_flat, 2, idx00.expand(b, c, hg * wg)).view(b, c, hg, wg)
    v01 = torch.gather(fmap_flat, 2, idx01.expand(b, c, hg * wg)).view(b, c, hg, wg)
    v10 = torch.gather(fmap_flat, 2, idx10.expand(b, c, hg * wg)).view(b, c, hg, wg)
    v11 = torch.gather(fmap_flat, 2, idx11.expand(b, c, hg * wg)).view(b, c, hg, wg)

    # apply weights and masks
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
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = int(num_levels)
        self.radius = int(radius)

        # correlation pyramid 
        corr = CorrBlockONNX.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid = [corr]
        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        # [B, 2, H, W] -> [B, H, W, 2]
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        dx = torch.arange(-r, r + 1, device=coords.device, dtype=coords.dtype)
        dy = torch.arange(-r, r + 1, device=coords.device, dtype=coords.dtype)
        # torch.stack(torch.meshgrid(dy, dx), axis=-1)
        yy, xx = torch.meshgrid(dy, dx)
        delta = torch.stack([yy, xx], dim=-1)  # [y, x] 

        out_pyramid = []
        for i in range(self.num_levels):
            corr_lvl = self.corr_pyramid[i]

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / (2 ** i)
            delta_lvl = delta.reshape(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            # sampled = bilinear_sampler(corr_lvl, coords_lvl)
            xgrid = coords_lvl[..., 0]
            ygrid = coords_lvl[..., 1]
            sampled = _bilinear_sample(corr_lvl, xgrid, ygrid)
            sampled = sampled.reshape(batch, h1, w1, -1)
            out_pyramid.append(sampled)

        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1_flat = fmap1.reshape(batch, dim, ht * wd)
        fmap2_flat = fmap2.reshape(batch, dim, ht * wd)

        corr = torch.matmul(fmap1_flat.transpose(1, 2), fmap2_flat)
        corr = corr.reshape(batch, ht, wd, 1, ht, wd)
        scale = 1.0 / float(math.sqrt(dim))
        return corr * scale

class RAFTModelWrapper(RAFT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, image1, image2, iters=20, flow_init=None, upsample=True, test_mode=True):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlockONNX(fmap1, fmap2, radius=self.args.corr_radius)
            #corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
