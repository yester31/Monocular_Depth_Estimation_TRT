# by yhpark 2025-7-24
import os
os.environ['XFORMERS_DISABLED'] = '1'   # Disable xformers
import torch

from moge.model.v2 import MoGeModel
from typing import *
import torch.nn.functional as F


class MoGeModelWrapper(MoGeModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, image: torch.Tensor, num_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        original_interpolate = F.interpolate

        def safe_interpolate(*args, **kwargs):
            kwargs['antialias'] = False
            return original_interpolate(*args, **kwargs)

        F.interpolate = safe_interpolate
        try:
            # 원래 MoGeModel의 forward 실행
            out = super().forward(image, num_tokens)
        finally:
            # 항상 원래 interpolate로 복구
            F.interpolate = original_interpolate

        return out




def main():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # Load model
    encoder = 'vits' # 'vitl' or 'vitb', 'vits'
    model_name = f"moge-2-{encoder}-normal"
    checkpoint = f"{cur_dir}/MoGe/checkpoint/{model_name}/model.pt"
    wrapped_model = MoGeModelWrapper.from_pretrained(checkpoint).eval()

if __name__ == '__main__':
    main()