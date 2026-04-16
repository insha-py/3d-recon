import numpy as np


def save_poses(poses, output_path):

    np.save(output_path, poses)


def load_poses(path):

    return np.load(path)
