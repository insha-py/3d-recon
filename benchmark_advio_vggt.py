import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.datasets.advio import ADVIO_SEQUENCE_TAGS, interpolate_poses, load_advio_iphone_sequence


def load_predicted_pose_dir(path):
    path = Path(path)
    poses = {}
    for pose_file in sorted(path.glob("*_extrinsic.txt")):
        extrinsic = np.loadtxt(pose_file, dtype=np.float64).reshape(4, 4)
        c2w = np.linalg.inv(extrinsic)
        key = pose_file.stem.replace("_extrinsic", "")
        poses[key] = c2w
    for pose_file in sorted(path.glob("*.txt")):
        if pose_file.name.endswith("_intrinsic.txt") or pose_file.name.endswith("_extrinsic.txt"):
            continue
        key = pose_file.stem
        if key not in poses:
            poses[key] = np.loadtxt(pose_file, dtype=np.float64).reshape(4, 4)
    return poses


def load_frame_metadata(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for parts in reader:
            if len(parts) < 2:
                continue
            rows.append((parts[0].strip(), float(parts[1])))
    return rows


def umeyama_alignment(src_points, dst_points):
    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    covariance = (dst_centered.T @ src_centered) / len(src_points)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    src_var = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = np.trace(np.diag(singular_values) @ correction) / max(src_var, 1e-12)
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def apply_similarity(poses, scale, rotation, translation):
    transformed = poses.copy()
    transformed[:, :3, :3] = rotation @ transformed[:, :3, :3]
    transformed[:, :3, 3] = (scale * (rotation @ transformed[:, :3, 3].T)).T + translation
    return transformed


def rotation_error_deg(a_pose, b_pose):
    rel = Rotation.from_matrix(a_pose[:3, :3]).inv() * Rotation.from_matrix(b_pose[:3, :3])
    return np.linalg.norm(rel.as_rotvec()) * 180.0 / np.pi


def translation_error(a_pose, b_pose):
    return np.linalg.norm(a_pose[:3, 3] - b_pose[:3, 3])


def relative_pose_errors(pred_poses, gt_poses):
    if len(pred_poses) < 2:
        return np.array([]), np.array([])
    rot_errors = []
    trans_errors = []
    for index in range(1, len(pred_poses)):
        pred_rel = np.linalg.inv(pred_poses[index - 1]) @ pred_poses[index]
        gt_rel = np.linalg.inv(gt_poses[index - 1]) @ gt_poses[index]
        rot_errors.append(rotation_error_deg(pred_rel, gt_rel))
        trans_errors.append(translation_error(pred_rel, gt_rel))
    return np.asarray(rot_errors), np.asarray(trans_errors)


def summarize_errors(rot_errors, trans_errors, ate):
    summary = {"count": int(len(rot_errors))}
    if len(rot_errors) == 0:
        summary.update({"rotation_mean_deg": None, "translation_mean": None, "ate_rmse": None})
        return summary
    summary.update(
        {
            "rotation_mean_deg": float(np.mean(rot_errors)),
            "rotation_median_deg": float(np.median(rot_errors)),
            "translation_mean": float(np.mean(trans_errors)),
            "translation_median": float(np.median(trans_errors)),
            "ate_rmse": float(np.sqrt(np.mean(ate ** 2))),
        }
    )
    return summary


def compute_turn_mask(frame_timestamps, gyroscope, threshold_rad_s):
    gyro_times = gyroscope[:, 0]
    gyro_mag = np.linalg.norm(gyroscope[:, 1:4], axis=1)
    sampled = np.interp(frame_timestamps, gyro_times, gyro_mag)
    return sampled >= threshold_rad_s


def compute_vertical_motion_mask(poses, threshold):
    if len(poses) < 2:
        return np.zeros(len(poses), dtype=bool)
    dz = np.abs(np.diff(poses[:, 2, 3], prepend=poses[0, 2, 3]))
    return dz >= threshold


def subset_summary(pred_poses, gt_poses, mask):
    if mask.sum() < 2:
        return {"count": int(mask.sum()), "rotation_mean_deg": None, "translation_mean": None, "ate_rmse": None}
    masked_pred = pred_poses[mask]
    masked_gt = gt_poses[mask]
    ate = np.linalg.norm(masked_pred[:, :3, 3] - masked_gt[:, :3, 3], axis=1)
    rre, rte = relative_pose_errors(masked_pred, masked_gt)
    return summarize_errors(rre, rte, ate)


def build_run_metrics(predicted, frame_rows, gt_frame_poses, sequence, original_frame_count):
    names = [name for name, _ in frame_rows if name.rsplit(".", 1)[0] in predicted]
    if not names:
        return {
            "frames": 0,
            "count": 0,
            "rotation_mean_deg": None,
            "rotation_median_deg": None,
            "translation_mean": None,
            "translation_median": None,
            "ate_rmse": None,
            "frame_reduction_percent": None,
            "gpu_frame_proxy_ratio": None,
            "gpu_attention_proxy_ratio": None,
            "similarity_scale": None,
            "turn_segments": {"count": 0, "rotation_mean_deg": None, "translation_mean": None, "ate_rmse": None},
            "vertical_motion_segments": {"count": 0, "rotation_mean_deg": None, "translation_mean": None, "ate_rmse": None},
            "dataset_tags": sequence.tags,
            "crowd_proxy": sequence.tags.get("people"),
        }
    timestamps = np.asarray([timestamp for name, timestamp in frame_rows if name.rsplit(".", 1)[0] in predicted], dtype=np.float64)
    pred_poses = np.asarray([predicted[name.rsplit(".", 1)[0]] for name in names], dtype=np.float64)
    gt_poses = gt_frame_poses[[index for index, (name, _) in enumerate(frame_rows) if name.rsplit(".", 1)[0] in predicted]]

    scale, rotation, translation = umeyama_alignment(pred_poses[:, :3, 3], gt_poses[:, :3, 3])
    aligned_pred = apply_similarity(pred_poses, scale, rotation, translation)

    ate = np.linalg.norm(aligned_pred[:, :3, 3] - gt_poses[:, :3, 3], axis=1)
    rre, rte = relative_pose_errors(aligned_pred, gt_poses)
    summary = summarize_errors(rre, rte, ate)
    summary["frames"] = int(len(aligned_pred))
    summary["frame_reduction_percent"] = float(100.0 * (1.0 - len(aligned_pred) / max(original_frame_count, 1)))
    summary["gpu_frame_proxy_ratio"] = float(len(aligned_pred) / max(original_frame_count, 1))
    summary["gpu_attention_proxy_ratio"] = float((len(aligned_pred) ** 2) / max(original_frame_count ** 2, 1))
    summary["similarity_scale"] = float(scale)

    turn_mask = compute_turn_mask(timestamps, sequence.gyroscope, threshold_rad_s=1.0)
    vertical_mask = compute_vertical_motion_mask(gt_poses, threshold=0.03)
    summary["turn_segments"] = subset_summary(aligned_pred, gt_poses, turn_mask)
    summary["vertical_motion_segments"] = subset_summary(aligned_pred, gt_poses, vertical_mask)
    summary["dataset_tags"] = sequence.tags
    summary["crowd_proxy"] = sequence.tags.get("people")
    return summary


def main(args):
    sequence = load_advio_iphone_sequence(args.advio_root)

    baseline_rows = load_frame_metadata(args.baseline_frames_csv)
    baseline_gt = interpolate_poses(
        np.asarray([timestamp for _, timestamp in baseline_rows], dtype=np.float64),
        sequence.ground_truth_timestamps,
        sequence.ground_truth_poses,
    )

    baseline_pred = load_predicted_pose_dir(args.baseline_pose_dir)
    results = {
        "sequence_id": sequence.sequence_id,
        "sequence_tags": ADVIO_SEQUENCE_TAGS.get(sequence.sequence_id, {}),
        "baseline": build_run_metrics(baseline_pred, baseline_rows, baseline_gt, sequence, len(baseline_rows)),
    }

    if args.imu_guided_pose_dir and args.imu_guided_frames_csv:
        guided_rows = load_frame_metadata(args.imu_guided_frames_csv)
        guided_gt = interpolate_poses(
            np.asarray([timestamp for _, timestamp in guided_rows], dtype=np.float64),
            sequence.ground_truth_timestamps,
            sequence.ground_truth_poses,
        )
        guided_pred = load_predicted_pose_dir(args.imu_guided_pose_dir)
        results["imu_guided"] = build_run_metrics(guided_pred, guided_rows, guided_gt, sequence, len(baseline_rows))
        results["imu_vs_baseline"] = {
            "rotation_mean_delta_deg": None
            if results["baseline"]["rotation_mean_deg"] is None or results["imu_guided"]["rotation_mean_deg"] is None
            else results["imu_guided"]["rotation_mean_deg"] - results["baseline"]["rotation_mean_deg"],
            "translation_mean_delta": None
            if results["baseline"]["translation_mean"] is None or results["imu_guided"]["translation_mean"] is None
            else results["imu_guided"]["translation_mean"] - results["baseline"]["translation_mean"],
            "ate_rmse_delta": None
            if results["baseline"]["ate_rmse"] is None or results["imu_guided"]["ate_rmse"] is None
            else results["imu_guided"]["ate_rmse"] - results["baseline"]["ate_rmse"],
            "gpu_frame_proxy_delta": None
            if results["baseline"]["gpu_frame_proxy_ratio"] is None or results["imu_guided"]["gpu_frame_proxy_ratio"] is None
            else results["imu_guided"]["gpu_frame_proxy_ratio"] - results["baseline"]["gpu_frame_proxy_ratio"],
            "gpu_attention_proxy_delta": None
            if results["baseline"]["gpu_attention_proxy_ratio"] is None or results["imu_guided"]["gpu_attention_proxy_ratio"] is None
            else results["imu_guided"]["gpu_attention_proxy_ratio"] - results["baseline"]["gpu_attention_proxy_ratio"],
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Saved benchmark report to {output_path}")
    print(f"Baseline ATE RMSE: {results['baseline']['ate_rmse']}")
    if "imu_guided" in results:
        print(f"IMU-guided ATE RMSE: {results['imu_guided']['ate_rmse']}")
        print(f"GPU frame proxy ratio: {results['imu_guided']['gpu_frame_proxy_ratio']}")
        print(f"GPU attention proxy ratio: {results['imu_guided']['gpu_attention_proxy_ratio']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark baseline and IMU-guided VGGT runs on ADVIO.")
    parser.add_argument("--advio-root", required=True, help="Path to one ADVIO sequence root, e.g. data/advio-01.")
    parser.add_argument("--baseline-frames-csv", required=True, help="CSV from prepare_advio_vggt_dataset baseline metadata.")
    parser.add_argument("--baseline-pose-dir", required=True, help="VGGT output directory with baseline pose txt files.")
    parser.add_argument("--imu-guided-frames-csv", help="CSV from prepare_advio_vggt_dataset imu_guided metadata.")
    parser.add_argument("--imu-guided-pose-dir", help="VGGT output directory with IMU-guided pose txt files.")
    parser.add_argument("--output-json", required=True, help="Where to save the benchmark JSON report.")
    main(parser.parse_args())
