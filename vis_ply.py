import open3d as o3d
import os 
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
'''
conda install -c conda-forge libstdcxx-ng
export XDG_SESSION_TYPE=x11
export GDK_BACKEND=x11
'''
def vis_point_cloud(point_path):
    pcd = o3d.io.read_point_cloud(point_path)
    # pcd.paint_uniform_color([1, 0, 0])  # red
    # pcd.paint_uniform_color([0, 1, 0])  # green
    o3d.visualization.draw_geometries([pcd])

point_path = f"{CUR_DIR}/VGGT/results/example_vggt_518x518_trt2.ply"

vis_point_cloud(point_path)
