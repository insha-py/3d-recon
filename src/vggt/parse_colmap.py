import numpy as np
from pathlib import Path
import sys

# Add path to colmap reader
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from read_write_model import read_model


def qvec_to_rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = qvec

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def parse_colmap_sparse(sparse_dir):
    """
    Parses COLMAP sparse reconstruction folder.

    Returns:
        points_xyz: (N,3)
        points_rgb: (N,3)
        poses: (N_images,4,4)
        image_names: list
    """

    sparse_dir = Path(sparse_dir)

    cameras, images, points3D = read_model(str(sparse_dir))

    # Parse point cloud
    xyz = []
    rgb = []

    for point in points3D.values():
        xyz.append(point.xyz)
        rgb.append(point.rgb)

    xyz = np.array(xyz)
    rgb = np.array(rgb)

    # Parse camera poses
    poses = []
    image_names = []

    for image in images.values():

        R = qvec_to_rotmat(image.qvec)
        t = image.tvec

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        # Convert from world-to-camera → camera-to-world
        T = np.linalg.inv(T)

        poses.append(T)
        image_names.append(image.name)

    poses = np.array(poses)

    return xyz, rgb, poses, image_names
