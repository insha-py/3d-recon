import open3d as o3d
import numpy as np


def create_pointcloud(xyz, rgb):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    return pcd


def save_pointcloud(pcd, output_path):

    o3d.io.write_point_cloud(str(output_path), pcd)


def load_pointcloud(path):

    return o3d.io.read_point_cloud(str(path))
