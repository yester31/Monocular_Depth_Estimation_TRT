# by yhpark 2025-8-9
import sys
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import time

sys.path.insert(1, os.path.join(sys.path[0], "NeuFlow_v2"))

from NeuFlow_v2.NeuFlow.neuflow import NeuFlow
from NeuFlow_v2.NeuFlow.backbone_v7 import ConvBlock
from NeuFlow_v2.data_utils import flow_viz
from wrapper import NeuFlowModelWrapper

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
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # height, width = image.shape[:2]
    if new_size is not None :
        image = cv2.resize(image, new_size)

    image = np.transpose(image, (2, 0, 1))

    image = np.ascontiguousarray(image).astype(np.float32)
    # [C, H, W] -> [1, C, H, W]
    image = np.expand_dims(image, axis=0)

    image = torch.from_numpy(image).to(DEVICE)
    return image

def fuse_conv_and_bn(conv, bn):
        """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
        fusedconv = (
            torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

def main():

    save_dir_path = os.path.join(CUR_DIR, 'results', 'pytorch')
    os.makedirs(save_dir_path, exist_ok=True)

    #model = NeuFlow().to(DEVICE)
    model = NeuFlowModelWrapper().to(DEVICE)
    checkpoint = torch.load(f'{CUR_DIR}/NeuFlow_v2/neuflow_mixed.pth', map_location='cuda')
    model.load_state_dict(checkpoint['model'], strict=True)
    
    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
            delattr(m, "norm1")  # remove batchnorm
            delattr(m, "norm2")  # remove batchnorm
            m.forward = m.forward_fuse  # update forward
    
    model.eval()
    model.half()

    image_width, image_height = 512, 288
    #image_width, image_height = 768, 432
    model.init_bhwd(1, image_height, image_width, 'cuda', True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{CUR_DIR}/results/video_neuflow_pytorch.mp4", fourcc, 20, (640, 480))

    with torch.no_grad():
        images = glob.glob(os.path.join(f'{CUR_DIR}/../video_frames', '*.png')) + \
                 glob.glob(os.path.join(f'{CUR_DIR}/../video_frames', '*.jpg'))
        dur_time = 0
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            # image1 = load_image(imfile1).half()
            # image2 = load_image(imfile2).half()
            # image1 = load_image0(imfile1,(image_width, image_height))
            # image2 = load_image0(imfile2,(image_width, image_height))
            image1 = load_image0(imfile1,(image_width, image_height)).half()
            image2 = load_image0(imfile2,(image_width, image_height)).half()

            begin = time.time()
            flow = model(image1, image2)# [[1,2,h,w]]
            flow = flow[-1] # [1,2,h,w]
            flow = flow[0]# [2,h,w]
            torch.cuda.synchronize()
            dur_time += time.time() - begin

            flow = flow.permute(1,2,0).cpu().numpy()
            flow = flow_viz.flow_to_image(flow)
            vis_flow = cv2.cvtColor(flow, cv2.COLOR_RGB2BGR)

            image1 = image1[0].permute(1,2,0).cpu().numpy() 
            image1 = (image1 * 255).astype(np.uint8) 

            # map flow to rgb image
            img_flo = np.concatenate([image1, vis_flow], axis=0)

            cv2.imshow('image', img_flo/255.0)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            output_path = f'{save_dir_path}/{os.path.splitext(os.path.basename(imfile1))[0]}_optical_flow.jpg'
            cv2.imwrite(output_path, img_flo)
            
            frame = cv2.resize(vis_flow, (640, 480))
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
