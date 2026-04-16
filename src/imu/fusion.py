import numpy as np
from scipy.spatial.transform import Rotation


def c2w_to_extrinsic(c2w_poses):
    c2w_poses = np.asarray(c2w_poses, dtype=np.float64)
    if c2w_poses.ndim == 2:
        return np.linalg.inv(c2w_poses)
    return np.asarray([np.linalg.inv(pose) for pose in c2w_poses], dtype=np.float64)


def extrinsic_to_c2w(extrinsics):
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.ndim == 2:
        return np.linalg.inv(extrinsics)
    return np.asarray([np.linalg.inv(pose) for pose in extrinsics], dtype=np.float64)


def _slerp_like_rotation_from_imu(imu_rotation, vision_rotation, vision_weight):
    vision_weight = float(np.clip(vision_weight, 0.0, 1.0))
    imu_rot = Rotation.from_matrix(imu_rotation)
    vis_rot = Rotation.from_matrix(vision_rotation)
    residual = imu_rot.inv() * vis_rot
    fused = imu_rot * Rotation.from_rotvec(residual.as_rotvec() * vision_weight)
    return fused.as_matrix()


def fuse_camera_extrinsics_with_imu(extrinsics_vggt, imu_c2w_poses, vision_weight=0.35, keep_translation="vision"):
    extrinsics_vggt = np.asarray(extrinsics_vggt, dtype=np.float64)
    imu_c2w_poses = np.asarray(imu_c2w_poses, dtype=np.float64)

    vision_c2w = extrinsic_to_c2w(extrinsics_vggt)
    fused_c2w = vision_c2w.copy()

    for index in range(len(fused_c2w)):
        fused_c2w[index, :3, :3] = _slerp_like_rotation_from_imu(
            imu_c2w_poses[index, :3, :3],
            vision_c2w[index, :3, :3],
            vision_weight=vision_weight,
        )
        if keep_translation == "imu":
            fused_c2w[index, :3, 3] = imu_c2w_poses[index, :3, 3]
        else:
            fused_c2w[index, :3, 3] = vision_c2w[index, :3, 3]

    return c2w_to_extrinsic(fused_c2w)
