import torch
import os
import numpy as np
import argparse
import sys

sys.path.insert(1, os.path.join(sys.path[0], "LiteVGGT_repo"))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.eval_utils import load_image_file_crop
from vggt.utils.geometry import unproject_depth_map_to_point_map

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MDET] using device: {DEVICE}")


def main():
    save_dir_path = os.path.join(CUR_DIR, "results")
    os.makedirs(save_dir_path, exist_ok=True)

    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    model = VGGT().to(DEVICE)

    ckpt_path = f"{CUR_DIR}/checkpoints/te_dict.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(torch.bfloat16)
    model.eval()
    print("Model loaded")

    img_path = f"{CUR_DIR}/../data/example.jpg"
    print(img_path)

    # (H, W, 3) 0-1
    img = load_image_file_crop(img_path)
    # print(img.shape)
    # 3 h w 0-1
    image_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)
    print(f"✅ images: {image_tensor.shape}")

    patch_width = image_tensor.shape[-1] // 14
    patch_height = image_tensor.shape[-2] // 14
    model.update_patch_dimensions(patch_width, patch_height)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
            aggregated_tokens_list, patch_start_idx = model.aggregator(image_tensor)

        with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            w2c_pre, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, image_tensor.shape[-2:]
            )

            depth_map, depth_conf = model.depth_head(
                aggregated_tokens_list, image_tensor, patch_start_idx
            )

        # # numpy (S, H, W, 3) word
        # points_3d = unproject_depth_map_to_point_map(
        #     depth_map.squeeze(0), w2c_pre.squeeze(0), intrinsic.squeeze(0)
        # )

        # depth_map = depth_map.squeeze(0)  # [1,518,518,1]

        # depth_map = depth_map.permute(0, 3, 1, 2)  # [1, 1, 518, 518]
        # depth_map = F.interpolate(
        #     depth_map,
        #     size=(target_size, target_size),
        #     mode="bilinear",
        #     align_corners=False,
        # )  # [1, 1, 1024, 1024]
        # depth_map = depth_map[
        #     ..., int(y1) : int(y2), int(x1) : int(x2)
        # ]  # remove paddings # [1, 1, 768, 1024]
        # depth_map = depth_map.permute(0, 2, 3, 1)  # [1, 768, 1024, 1]
        # depth_map = depth_map.squeeze(0)

        depth_map = depth_map.cpu().numpy()
        depth = np.squeeze(depth_map)
        print(f"[MDET] max : {depth.max():0.5f} , min : {depth.min():0.5f}")

        print("[MDET] Generate color depth image")
        inverse_depth = 1 / depth
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu + 1e-6
        )

        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
        color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)
        color_depth_bgr = cv2.resize(color_depth_bgr, (width, height), cv2.INTER_LINEAR)

        # save colored depth image
        output_file_depth = os.path.join(
            save_dir_path,
            os.path.splitext(image_file_name)[0] + f"_LiteVGGT_torch_depth.jpg",
        )
        cv2.imwrite(output_file_depth, color_depth_bgr)
        print("done!")


if __name__ == "__main__":
    main()
