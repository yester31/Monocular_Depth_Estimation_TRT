"""Open a .ply point cloud in an Open3D window.

    python tools/retired/vis_ply.py

The path below is edited by hand to whatever cloud is being looked at; the
default points at a VGGT result under the repository root.
"""

import open3d as o3d
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

point_path = f"{ROOT}/VGGT/results/example_vggt_518x518_trt2.ply"

vis_point_cloud(point_path)
