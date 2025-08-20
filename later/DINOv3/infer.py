from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os 
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])

REPO_DIR = f'{CUR_DIR}/dinov3'
weights_path = f'{REPO_DIR}/pretrained/dinov3_vit7b16_synthmix_dpt_head-02040be1.pth'
backbone_weights_path = f'{REPO_DIR}/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
depther = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_dd', source="local", weights=weights_path, backbone_weights=backbone_weights_path)

img_size = 1024
img = get_img()
transform = make_transform(img_size)
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        batch_img = batch_img
        depths = depther(batch_img)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.imshow(depths[0,0].cpu(), cmap=colormaps["Spectral"])
plt.axis("off")