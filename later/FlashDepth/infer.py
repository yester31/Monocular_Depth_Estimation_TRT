# by yhpark 2025-8-13
import os
import sys
import cv2
import tqdm
import numpy as np
import torch
import time
from matplotlib import pyplot as plt

from omegaconf import OmegaConf

sys.path.insert(1, os.path.join(sys.path[0], "FlashDepth"))
from wrapper import FlashDepthModelWrapper
from FlashDepth.train import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

def constrain_to_multiple_of(x, min_val=0, max_val=None, ensure_multiple_of=14):
    y = (np.round(x / ensure_multiple_of) * ensure_multiple_of).astype(int)

    if max_val is not None and y > max_val:
        y = (np.floor(x / ensure_multiple_of) * ensure_multiple_of).astype(int)

    if y < min_val:
        y = (np.ceil(x / ensure_multiple_of) * ensure_multiple_of).astype(int)

    return y

def load_image(image_path, new_size=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]

    if max(h, w) > 2044: # set max long side to 2044
        scale = 2044 / max(h, w)
        res = (int(w * scale), int(h * scale))
    else:
        res = (w, h)

    if new_size is not None:
        res = new_size

    target_w, target_h = res  # Unpack width and height
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    image = np.clip(image, 0, 1)

    # resize sample (ensure_multiple_of 14)
    new_height = constrain_to_multiple_of(target_h, min_val=target_h)
    new_width = constrain_to_multiple_of(target_w, min_val=target_w)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # NormalizeImage
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))

    image = np.ascontiguousarray(image).astype(np.float32)
    # [C, H, W] -> [1, C, H, W]
    image = np.expand_dims(image, axis=0)

    image = torch.from_numpy(image)
    return image

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results', 'pytorch3')
    os.makedirs(save_dir_path, exist_ok=True)

    # MODEL
    print(f'[MDET] load model')
    config_dir_path = f"{CUR_DIR}/FlashDepth/configs/flashdepth"
    cfg = OmegaConf.load(f"{config_dir_path}/config.yaml")
    model = FlashDepthModelWrapper(**dict( 
        batch_size=cfg.training.batch_size, 
        hybrid_configs=cfg.hybrid_configs,
        training=False,
        **cfg.model,
    ))

    model = model.cpu()
    checkpoint_path = f'{config_dir_path}/iter_43002.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(model_state_dict, strict=True)
    model = model.to(DEVICE)
    model.eval()

    image_dir_name = 'video_frames'
    image_dir = os.path.join(CUR_DIR, '..', image_dir_name)
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]
    image_paths = [os.path.join(image_dir, fname) for fname in sorted(os.listdir(image_dir)) if os.path.splitext(fname)[1].lower() in valid_exts]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{CUR_DIR}/results/video_flashdepth_pytorch3.mp4", fourcc, 20, (640, 480))
    cmap = plt.get_cmap("turbo")

    print(f'[MDET] run start')
    dur_time = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for image_path in tqdm(image_paths):
                image = load_image(image_path, (518, 518)).to(DEVICE)

                begin = time.time()
                pred_depth = model(image)
                torch.cuda.synchronize()
                dur_time += time.time() - begin

                depth = pred_depth.detach().cpu().float()
                min_val = depth.min()
                max_val = depth.max()
                if max_val - min_val == 0:
                    depth_normalized = torch.zeros_like(depth)
                else:
                    depth_normalized = (depth - min_val) / (max_val - min_val)

                depth_normalized = depth_normalized.squeeze().numpy()  # (H, W)
                color_depth = (cmap(depth_normalized)[..., :3] * 255).astype(np.uint8)

                output_path = f'{save_dir_path}/{os.path.splitext(os.path.basename(image_path))[0]}_depth.jpg'
                cv2.imwrite(output_path, color_depth)
                
                frame = cv2.resize(color_depth, (640, 480))
                out.write(frame)

            out.release()
            cv2.destroyAllWindows()
            
            iteration = len(image_paths) - 1
            print(f'[MDET] {iteration} iterations time: {dur_time:.4f} [sec]')
            avg_time = dur_time / iteration
            print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
            print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')


if __name__ == '__main__':
    main()
