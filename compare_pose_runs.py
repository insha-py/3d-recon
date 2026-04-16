import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_pose_dir(path):
    path = Path(path)
    poses = {}
    for pose_file in sorted(path.glob("*_extrinsic.txt")):
        extrinsic = np.loadtxt(pose_file, dtype=np.float64).reshape(4, 4)
        c2w = np.linalg.inv(extrinsic)
        poses[pose_file.stem.replace("_extrinsic", "")] = c2w
    return poses


def load_prior_dir(path):
    path = Path(path)
    poses = {}
    for pose_file in sorted(path.glob("*.txt")):
        poses[pose_file.stem] = np.loadtxt(pose_file, dtype=np.float64).reshape(4, 4)
    return poses


def relative_rotation_error_deg(a_pose, b_pose):
    rel = Rotation.from_matrix(a_pose[:3, :3]).inv() * Rotation.from_matrix(b_pose[:3, :3])
    return np.linalg.norm(rel.as_rotvec()) * 180.0 / np.pi


def translation_error(a_pose, b_pose):
    return np.linalg.norm(a_pose[:3, 3] - b_pose[:3, 3])


def summarize(name, errors):
    if not errors:
        print(f"{name}: no overlapping frames")
        return
    print(
        f"{name}: mean={np.mean(errors):.4f} median={np.median(errors):.4f} "
        f"max={np.max(errors):.4f} n={len(errors)}"
    )


def compare_to_reference(run_poses, reference_poses, label):
    rot_errors = []
    trans_errors = []
    for key in sorted(set(run_poses) & set(reference_poses)):
        rot_errors.append(relative_rotation_error_deg(run_poses[key], reference_poses[key]))
        trans_errors.append(translation_error(run_poses[key], reference_poses[key]))
    summarize(f"{label} rotation(deg)", rot_errors)
    summarize(f"{label} translation", trans_errors)


def main(args):
    baseline = load_pose_dir(args.baseline_dir)
    imu_guided = load_pose_dir(args.imu_guided_dir)

    print("Baseline vs IMU-guided")
    compare_to_reference(baseline, imu_guided, "run-to-run")

    if args.prior_dir:
        priors = load_prior_dir(args.prior_dir)
        print("\nBaseline vs IMU priors")
        compare_to_reference(baseline, priors, "baseline-to-prior")
        print("\nIMU-guided vs IMU priors")
        compare_to_reference(imu_guided, priors, "imu-guided-to-prior")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare VGGT pose outputs across two runs.")
    parser.add_argument("--baseline-dir", required=True, help="Folder with baseline *_extrinsic.txt pose files.")
    parser.add_argument("--imu-guided-dir", required=True, help="Folder with IMU-guided *_extrinsic.txt pose files.")
    parser.add_argument("--prior-dir", help="Optional folder with IMU prior c2w pose files.")
    main(parser.parse_args())
