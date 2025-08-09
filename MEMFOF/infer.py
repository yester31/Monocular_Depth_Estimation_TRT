# by yhpark 2025-8-9
import sys
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import time

sys.path.insert(1, os.path.join(sys.path[0], "memfof"))
from memfof.core.memfof import MEMFOF
from memfof.core.utils.flow_viz import flow_to_image

DEVICE = 'cuda'
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

AVAILABLE_MODELS = [
    "MEMFOF-Tartan",
    "MEMFOF-Tartan-T",
    "MEMFOF-Tartan-T-TSKH",
    "MEMFOF-Tartan-T-TSKH-kitti",
    "MEMFOF-Tartan-T-TSKH-sintel",
    "MEMFOF-Tartan-T-TSKH-spring",
]

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

    model = MEMFOF.from_pretrained("egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH")
    model.to(DEVICE)
    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{CUR_DIR}/results/video_memfof_pytorch.mp4", fourcc, 20, (640, 480))

    height, width = 288, 512, 
    with torch.no_grad():
        images = glob.glob(os.path.join(f'{CUR_DIR}/../video_frames', '*.png')) + \
                 glob.glob(os.path.join(f'{CUR_DIR}/../video_frames', '*.jpg'))
        dur_time = 0
        images = sorted(images)
        for imfile1, imfile2, imfile3 in zip(images[:-2], images[1:-1], images[2:]):
            image1 = load_image(imfile1) # [1, C, H, W]
            image2 = load_image(imfile2) # [1, C, H, W]
            image3 = load_image(imfile3) # [1, C, H, W]
            frames_tensor = torch.stack([image1, image2, image3], dim=1) # [1, 3, C, H, W]

            begin = time.time()
            output = model(frames_tensor)
            torch.cuda.synchronize()
            dur_time += time.time() - begin

            forward_flow = output["flow"]  # [[1, 2, 2, H, W]]
            forward_flow = forward_flow[-1]  # [1, 2, 2, H, W]
            forward_flow = forward_flow[:, 1]  # [1, 2, H, W]
            flow_vis = flow_to_image(
                forward_flow.squeeze(dim=0).permute(1, 2, 0).cpu().numpy(),
                convert_to_bgr = True,
                rad_min=0.02 * (height ** 2 + width ** 2) ** 0.5,
            )

            image = np.transpose(image2[0].cpu().numpy(), (1, 2, 0))
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_flo = np.concatenate([image_bgr, flow_vis], axis=0)

            cv2.imshow('image', img_flo/255.0)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            output_path = f'{save_dir_path}/{os.path.splitext(os.path.basename(imfile1))[0]}_optical_flow.jpg'
            cv2.imwrite(output_path, img_flo)
            
            frame = cv2.resize(flow_vis, (640, 480))
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
