# by yhpark 2025-8-9
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(1, os.path.join(sys.path[0], "MeFlow"))
from MeFlow.meflow.meflow import Model

class Attention1DOnnx(nn.Module):
    def __init__(self, in_channels, h_attention=True, r=8):
        super(Attention1DOnnx, self).__init__()

        self.h_attention = h_attention
        self.r = r

        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)

        # Window extraction via grouped conv (ONNX/TRT friendly)
        k = 2 * r + 1
        if self.h_attention:
            self.window_conv = nn.Conv2d(
                in_channels,
                in_channels * k,
                kernel_size=(1, k),
                padding=(0, r),
                groups=in_channels,
                bias=False,
            )
        else:
            self.window_conv = nn.Conv2d(
                in_channels,
                in_channels * k,
                kernel_size=(k, 1),
                padding=(r, 0),
                groups=in_channels,
                bias=False,
            )

        # Initialize (same as Transformer init for conv1x1) and one-hot for window_conv
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Overwrite window_conv weights to one-hot sampling kernels
        with torch.no_grad():
            self.window_conv.weight.zero_()
            k = 2 * self.r + 1
            for c_idx in range(in_channels):
                for k_idx in range(k):
                    out_ch = c_idx * k + k_idx
                    if self.h_attention:
                        self.window_conv.weight[out_ch, 0, 0, k_idx] = 1.0
                    else:
                        self.window_conv.weight[out_ch, 0, k_idx, 0] = 1.0

    def forward(self, feature, position=None, value=None):
        b, c, h, w = feature.size()
        feature = feature + position if position is not None else feature

        query = self.query_conv(feature)  # [B, C, H, W]
        key = self.key_conv(feature)      # [B, C, H, W]

        # Extract windows using grouped conv (no dynamic slice/concat)
        k = 2 * self.r + 1
        key_col = self.window_conv(key)                    # [B, C*K, H, W]
        val_col = self.window_conv(feature)                # [B, C*K, H, W]
        key_windows = key_col.view(b, c, k, h, w)          # [B, C, K, H, W]
        value_windows = val_col.view(b, c, k, h, w)

        # Scaled dot-product attention
        #scale = query.new_tensor(float(c) ** 0.5)
        # scale = torch.sqrt(query.new_tensor(c))
        scale = torch.sqrt(torch.tensor(c, device=query.device, dtype=query.dtype))
        scores = (query.unsqueeze(2) * key_windows).sum(dim=1, keepdim=True) / scale  # [B, 1, K, H, W]
        attention = torch.softmax(scores, dim=2)
        out = (attention * value_windows).sum(dim=2)        # [B, C, H, W]
        return out

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

class AlternateCorr1DOnnx:
    def __init__(self, fmap1, fmap2, fmap2_d1, fmap2_d2, radius=4, h_corr=True):
        self.radius = radius
        self.h_corr = h_corr

        if self.h_corr:
            self.fmap1 = fmap1
            self.fmap2 = fmap2
            self.fmap2_d1 = fmap2_d1
            self.fmap2_d2 = fmap2_d2
            b, c, h, w = self.fmap1.shape
            self.c = c
        else:
            self.fmap1 = fmap1
            self.fmap2 = fmap2
            self.fmap2_d1 = fmap2_d1
            self.fmap2_d2 = fmap2_d2
            b, c, h, w = self.fmap1.shape
            self.c = c

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        b, h, w, _ = coords.shape

        if self.h_corr:
            # horizontal offsets
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device, dtype=coords.dtype)
            # main scale sampling (use base y, shifted x)
            warped_list = []
            x_base = coords[..., 0]  # [B, H, W]
            y_base = coords[..., 1]
            for i in range(2 * r + 1):
                x_pos = x_base + dx[i]
                sampled = _bilinear_sample(self.fmap2, x_pos, y_base)  # [B, C, H, W]
                warped_list.append(sampled.unsqueeze(2))
            warped_fmap2 = torch.cat(warped_list, dim=2)  # [B, C, 2r+1, H, W]

            # downsampled scales: pick 2 from both ends
            delta_d = torch.stack((dx[0:2], dx[-2:]), dim=0).reshape(-1)  # 4 values

            warped_d1_list = []
            warped_d2_list = []
            for i in range(4):
                x_pos_d1 = x_base / 2.0 + delta_d[i]
                y_pos_d1 = y_base / 2.0
                x_pos_d2 = x_base / 4.0 + delta_d[i]
                y_pos_d2 = y_base / 4.0
                warped_d1_list.append(_bilinear_sample(self.fmap2_d1, x_pos_d1, y_pos_d1).unsqueeze(2))
                warped_d2_list.append(_bilinear_sample(self.fmap2_d2, x_pos_d2, y_pos_d2).unsqueeze(2))
            warped_fmap2_d1 = torch.cat(warped_d1_list, dim=2)  # [B, C, 4, H, W]
            warped_fmap2_d2 = torch.cat(warped_d2_list, dim=2)  # [B, C, 4, H, W]
        else:
            # vertical offsets
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device, dtype=coords.dtype)
            warped_list = []
            x_base = coords[..., 0]
            y_base = coords[..., 1]
            for i in range(2 * r + 1):
                y_pos = y_base + dy[i]
                sampled = _bilinear_sample(self.fmap2, x_base, y_pos)
                warped_list.append(sampled.unsqueeze(2))
            warped_fmap2 = torch.cat(warped_list, dim=2)

            delta_d = torch.stack((dy[0:2], dy[-2:]), dim=0).reshape(-1)
            warped_d1_list = []
            warped_d2_list = []
            for i in range(4):
                x_pos_d1 = x_base / 2.0
                y_pos_d1 = y_base / 2.0 + delta_d[i]
                x_pos_d2 = x_base / 4.0
                y_pos_d2 = y_base / 4.0 + delta_d[i]
                warped_d1_list.append(_bilinear_sample(self.fmap2_d1, x_pos_d1, y_pos_d1).unsqueeze(2))
                warped_d2_list.append(_bilinear_sample(self.fmap2_d2, x_pos_d2, y_pos_d2).unsqueeze(2))
            warped_fmap2_d1 = torch.cat(warped_d1_list, dim=2)
            warped_fmap2_d2 = torch.cat(warped_d2_list, dim=2)

        warped_fmap2 = torch.cat(
            (
                warped_fmap2_d2[:, :, 0:2],
                warped_fmap2_d1[:, :, 0:2],
                warped_fmap2,
                warped_fmap2_d1[:, :, -2:],
                warped_fmap2_d2[:, :, -2:],
            ),
            dim=2,
        )
        corr = (self.fmap1[:, :, None, :, :] * warped_fmap2).sum(dim=1)
        return corr / (self.c ** 0.5)

class MeFlowModelWrapper(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_h = Attention1DOnnx(self.feature_channels, h_attention=True, r=4)
        self.attn_v = Attention1DOnnx(self.feature_channels, h_attention=False, r=4)

    def forward(self, image1, image2, iters=20, flow_init=None, test_mode=True):
        """ Estimate optical flow between pair of frames """

        # torch.cuda.reset_max_memory_allocated()

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap2_attn_h = self.attn_h(fmap2)
        fmap2_attn_v = self.attn_v(fmap2) 


        fmap2_d1 = F.avg_pool2d(fmap2, 2, stride=2)
        fmap2_attn_h_d1 = self.attn_h(fmap2_d1)
        fmap2_attn_v_d1 = self.attn_v(fmap2_d1)

        fmap2_d2 = F.avg_pool2d(fmap2_d1, 2, stride=2)
        fmap2_attn_h_d2 = self.attn_h(fmap2_d2)
        fmap2_attn_v_d2 = self.attn_v(fmap2_d2)

        corr_fn_h = AlternateCorr1DOnnx(fmap1, fmap2_attn_v, fmap2_attn_v_d1, fmap2_attn_v_d2, radius=self.corr_radius, h_corr=True)
        corr_fn_v = AlternateCorr1DOnnx(fmap1, fmap2_attn_h, fmap2_attn_h_d1, fmap2_attn_h_d2, radius=self.corr_radius, h_corr=False)

        # run the context network
        cnet = self.cnet(image1)  # list of feature pyramid, low scale to high scale

        hdim = self.hidden_dim
        cdim = self.context_dim
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)  # 1/8 resolution or 1/4

        if flow_init is not None:  # flow_init is 1/8 resolution or 1/4
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()  # stop gradient
            flow = coords1 - coords0
            corr_h = corr_fn_h(coords1) # index 1D correlation volume
            corr_v = corr_fn_v(coords1) # index 1D correlation volume
            corr = torch.cat((corr_h, corr_v), dim=1)

            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow,
                                                         upsample=not test_mode or itr == iters - 1,
                                                         )

            coords1 = coords1 + delta_flow

            if test_mode:
                torch.cuda.empty_cache()
                # only upsample the last iteration
                if itr == iters - 1:
                    flow_up = self.learned_upflow(coords1 - coords0, up_mask)
                    # print('Max Allocated:', round(torch.cuda.max_memory_allocated(0)/1024**3,2), 'GB')

                    return coords1 - coords0, flow_up
            else:
                # upsample predictions
                flow_up = self.learned_upflow(coords1 - coords0, up_mask)
                flow_predictions.append(flow_up)

        return flow_predictions
