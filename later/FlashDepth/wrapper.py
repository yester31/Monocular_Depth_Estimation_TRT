# by yhpark 2025-8-13
import sys
import os
import torch

sys.path.insert(1, os.path.join(sys.path[0], "FlashDepth"))
from FlashDepth.flashdepth.model import FlashDepth

class FlashDepthModelWrapper(FlashDepth):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_mamba:
            self.mamba.start_new_sequence()

    @torch.no_grad()
    def forward(self, frame):

        B, C, H, W = frame.shape
        patch_h, patch_w = frame.shape[-2] // self.patch_size, frame.shape[-1] // self.patch_size

        dpt_features = self.get_dpt_features(frame, input_shape=(B,C,H,W)) 
        pred_depth = self.final_head(dpt_features, patch_h, patch_w)
        pred_depth = torch.clip(pred_depth, min=0)
    
        return pred_depth
