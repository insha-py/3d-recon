import open3d as o3d
import numpy as np


def visualize_pointcloud(pcd):

    o3d.visualization.draw_geometries([pcd])


def visualize_poses(poses):

    frames = []

    for pose in poses:

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(pose)

        frames.append(frame)

    o3d.visualization.draw_geometries(frames)


def visualize_reconstruction(pcd, poses):

    frames = []

    for pose in poses:

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(pose)

        frames.append(frame)

    o3d.visualization.draw_geometries([pcd, *frames])
