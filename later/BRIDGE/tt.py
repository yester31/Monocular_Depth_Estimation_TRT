import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], "Bridge"))
from bridge.dpt import Bridge 

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

save_dir_path = os.path.join(CUR_DIR, 'results')
os.makedirs(save_dir_path, exist_ok=True)

model = Bridge()
model.load_state_dict(torch.load(f'{CUR_DIR}/Bridge/checkpoints/bridge.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

file_path = f'{CUR_DIR}/../data/example.jpg'
# file_path = f'{CUR_DIR}/Bridge/examples/demo.png'
raw_img = cv2.imread(file_path)
# raw_img = cv2.resize(raw_img, (518, 518))

ori_shape = raw_img.shape[:2]
depth = model.infer_image(raw_img)  
print(f'[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}')

# ===================================================================
print('[MDET] Generate color depth image')
# visualization
depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth_normalized = depth_normalized.astype(np.uint8)

# Save as color-mapped "turbo" jpg image.
cmap = plt.get_cmap("turbo")
color_depth = (cmap(depth_normalized)[..., :3] * 255).astype(np.uint8)
color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)    

# save colored depth image
filename = os.path.splitext(os.path.basename(file_path))[0] 
output_file_depth = os.path.join(save_dir_path, filename + f'_torch.jpg')
cv2.imwrite(output_file_depth, color_depth_bgr)