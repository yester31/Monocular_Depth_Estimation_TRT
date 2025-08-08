# by yhpark 2025-8-9
import sys
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import time

sys.path.insert(1, os.path.join(sys.path[0], "MeFlow"))

from MeFlow.meflow.meflow import Model
from MeFlow.utils.utils import InputPadder
from MeFlow.utils.flow_viz import flow_to_image

from wrapper import MeFlowModelWrapper

DEVICE = 'cuda'
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def load_image(imfile):
    img = Image.open(imfile)

    # new_size = (int(img.width * 0.5), int(img.height * 0.5)) # (1024, 576)
    new_size = (int(img.width * 0.25), int(img.height * 0.25)) # (512, 288)
    img = img.resize(new_size, Image.LANCZOS)

    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_image0(image_path, new_size=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # height, width = image.shape[:2]
    if new_size is not None :
        image = cv2.resize(image, new_size)
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image).astype(np.float32)
    # [C, H, W] -> [1, C, H, W]
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).to(DEVICE)
    return image

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results', 'pytorch')
    os.makedirs(save_dir_path, exist_ok=True)

    model = Model(downsample_factor=8,
                 feature_channels=128,
                 corr_radius=4,
                 hidden_dim=128,
                 context_dim=128,
                 mixed_precision=True,
                 )

    checkpoint = torch.load(f'{CUR_DIR}/MeFlow/pretrained_models/sintel.pth')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=False)
    model.to(DEVICE)
    model.eval()

    # 프레임 정보
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{CUR_DIR}/results/output_video_pytorch.mp4", fourcc, 20, (640, 480))

    with torch.no_grad():
        images = glob.glob(os.path.join(f'{CUR_DIR}/../video_frames', '*.png')) + \
                 glob.glob(os.path.join(f'{CUR_DIR}/../video_frames', '*.jpg'))
        dur_time = 0
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            # image1 = load_image0(imfile1,(512, 288))
            # image2 = load_image0(imfile2,(512, 288))

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            begin = time.time()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

            image1 = image1[0].permute(1,2,0).cpu().numpy()
            flow_up = flow_up[0].permute(1,2,0).cpu().numpy()
            
            # map flow to rgb image
            vis_flow = flow_to_image(flow_up)
            img_flo = np.concatenate([image1, vis_flow], axis=0)

            cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            output_path = f'{save_dir_path}/{os.path.splitext(os.path.basename(imfile1))[0]}_optical_flow.jpg'
            cv2.imwrite(output_path, img_flo[:, :, [2,1,0]])
            
            frame = vis_flow[:, :, [2,1,0]]
            frame = cv2.resize(frame, (640, 480))
            out.write(frame)

        out.release()
        cv2.destroyAllWindows()
        
        iteration = len(images) - 1
        print(f'[MDET] {iteration} iterations time: {dur_time:.4f} [sec]')
        avg_time = dur_time / iteration
        print(f'[MDET] Average FPS: {1 / avg_time:.2f} [fps]')
        print(f'[MDET] Average inference time: {avg_time * 1000:.2f} [msec]')




if __name__ == '__main__':
    main()
