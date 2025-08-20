import torch
import imageio.v3 as iio
import sys 
import os 
sys.path.insert(1, os.path.join(sys.path[0], "co-tracker"))
from cotracker.utils.visualizer import Visualizer

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")

# 1. Download the video
url = 'https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4'
video_path = f'{CUR_DIR}/../video/video2.mp4'
frames = iio.imread(video_path, plugin="FFMPEG")  # plugin="pyav"
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(DEVICE)  # B T C H W

# 2. Load model
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(DEVICE)

# 3. Run Online CoTracker, the same model with a different API:
# Initialize online processing
grid_size = 10
cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)  

# 4. Process the video
for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
    pred_tracks, pred_visibility = cotracker(video_chunk=video[:, ind : ind + cotracker.step * 2])  # B T N 2,  B T N 1

# 5. Visualization 
vis = Visualizer(save_dir=f"{CUR_DIR}/results2", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)