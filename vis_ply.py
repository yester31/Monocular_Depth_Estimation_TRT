import open3d as o3d
import os 
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# conda install -c conda-forge libstdcxx-ng
# export XDG_SESSION_TYPE=x11
# export GDK_BACKEND=x11

def vis_point_cloud(point_path):
    pcd = o3d.io.read_point_cloud(point_path)
    # pcd.paint_uniform_color([1, 0, 0])  # 빨강
    # pcd.paint_uniform_color([0, 1, 0])  # 초록
    o3d.visualization.draw_geometries([pcd])

point_path1 = f"{CUR_DIR}/Prior_Depth_Anything/results/example_vggt2.ply"
point_path2 = f"{CUR_DIR}/VGGT/results/example_vggt.ply"
point_path3 = f"{CUR_DIR}/VGGT/results/example_vggt_trt.ply"
point_path4 = f"{CUR_DIR}/MoGe_2/MoGe/outputs/example/pointcloud.ply"
point_path5 = f"{CUR_DIR}/MoGe_2/results/example_vits_m2_torch_point_cloud.ply"
point_path6 = f"{CUR_DIR}/MoGe_2/results/example_vitb_m2_torch_point_cloud.ply"
point_path7 = f"{CUR_DIR}/MoGe_2/results/example_vitl_m2_torch_point_cloud.ply"
point_path8 = f"{CUR_DIR}/MoGe_2/results/example_vits_m2_trt_point_cloud.ply"
point_path9 = f"{CUR_DIR}/MoGe_2/results/7_vits_m2_trt_point_cloud.ply"
point_path10 = f"{CUR_DIR}/scene_output.ply"
point_path11 = f"{CUR_DIR}/Depth_Pro/results/example.ply"

#vis_point_cloud(point_path5)
#vis_point_cloud(point_path6)
vis_point_cloud(point_path11)
