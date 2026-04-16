import numpy as np
from scipy.spatial.transform import Rotation


def select_keyframes(poses, rotation_thresh_deg=4.0, translation_thresh_m=0.05, min_gap=1):
    if len(poses) == 0:
        return []
    if len(poses) == 1:
        return [0]

    selected = [0]
    last_index = 0

    for index in range(1, len(poses) - 1):
        if index - last_index < min_gap:
            continue

        prev_pose = poses[last_index]
        curr_pose = poses[index]

        rel_rot = Rotation.from_matrix(prev_pose[:3, :3]).inv() * Rotation.from_matrix(curr_pose[:3, :3])
        rel_angle_deg = np.linalg.norm(rel_rot.as_rotvec()) * 180.0 / np.pi
        rel_translation = np.linalg.norm(curr_pose[:3, 3] - prev_pose[:3, 3])

        rotation_active = rotation_thresh_deg is not None and rotation_thresh_deg > 0
        translation_active = translation_thresh_m is not None and translation_thresh_m > 0

        keep_for_rotation = rotation_active and rel_angle_deg >= rotation_thresh_deg
        keep_for_translation = translation_active and rel_translation >= translation_thresh_m

        if keep_for_rotation or keep_for_translation:
            selected.append(index)
            last_index = index

    if selected[-1] != len(poses) - 1:
        selected.append(len(poses) - 1)

    return selected
