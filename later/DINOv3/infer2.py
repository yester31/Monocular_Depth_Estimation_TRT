import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import os 
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from PIL import Image
import torch
from torchvision import transforms


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

img_size = 1024
img = get_img()
transform = make_transform(img_size)

REPO_DIR = f'{CUR_DIR}/dinov3'
weights_path = f'{REPO_DIR}/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=weights_path).to(DEVICE)

with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        batch_img = batch_img.to(DEVICE)
        outputs = dinov3_vits16(batch_img)

print("Pooled output shape:", outputs.shape)