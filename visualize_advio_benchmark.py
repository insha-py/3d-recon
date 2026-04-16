import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark_advio_vggt import (
    apply_similarity,
    load_frame_metadata,
    load_predicted_pose_dir,
    umeyama_alignment,
)
from src.datasets.advio import interpolate_poses, load_advio_iphone_sequence


def align_predicted_to_gt(predicted, frame_rows, gt_poses):
    names = [name for name, _ in frame_rows if name.rsplit(".", 1)[0] in predicted]
    if not names:
        return np.empty((0, 4, 4)), np.empty((0, 4, 4)), np.empty((0,))

    timestamps = np.asarray([timestamp for name, timestamp in frame_rows if name.rsplit(".", 1)[0] in predicted], dtype=np.float64)
    pred_poses = np.asarray([predicted[name.rsplit(".", 1)[0]] for name in names], dtype=np.float64)
    gt_selected = gt_poses[[index for index, (name, _) in enumerate(frame_rows) if name.rsplit(".", 1)[0] in predicted]]

    scale, rotation, translation = umeyama_alignment(pred_poses[:, :3, 3], gt_selected[:, :3, 3])
    aligned_pred = apply_similarity(pred_poses, scale, rotation, translation)
    return aligned_pred, gt_selected, timestamps


def make_trajectory_plot(output_path, gt_baseline, baseline_pred, gt_guided, guided_pred):
    plt.figure(figsize=(8, 6))
    if len(gt_baseline):
        plt.plot(gt_baseline[:, 0, 3], gt_baseline[:, 1, 3], label="Ground truth", linewidth=2)
    if len(baseline_pred):
        plt.plot(baseline_pred[:, 0, 3], baseline_pred[:, 1, 3], label="Baseline proxy", alpha=0.85)
    if len(guided_pred):
        plt.plot(guided_pred[:, 0, 3], guided_pred[:, 1, 3], label="IMU-guided proxy", alpha=0.85)
        plt.scatter(gt_guided[:, 0, 3], gt_guided[:, 1, 3], s=12, label="IMU-guided frame positions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("ADVIO trajectory (top-down)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_gyro_plot(output_path, sequence, baseline_times, guided_times):
    gyro_times = sequence.gyroscope[:, 0]
    gyro_mag = np.linalg.norm(sequence.gyroscope[:, 1:4], axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(gyro_times, gyro_mag, label="Gyro magnitude", linewidth=1)
    if len(baseline_times):
        plt.vlines(baseline_times, 0, np.percentile(gyro_mag, 95), color="tab:gray", alpha=0.12, label="Baseline frames")
    if len(guided_times):
        plt.vlines(guided_times, 0, np.percentile(gyro_mag, 95), color="tab:red", alpha=0.35, label="IMU-guided frames")
    plt.xlabel("time (s)")
    plt.ylabel("rad/s")
    plt.title("Gyroscope magnitude and selected frames")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_error_plot(output_path, baseline_times, baseline_pred, baseline_gt, guided_times, guided_pred, guided_gt):
    plt.figure(figsize=(10, 4))
    if len(baseline_pred):
        baseline_err = np.linalg.norm(baseline_pred[:, :3, 3] - baseline_gt[:, :3, 3], axis=1)
        plt.plot(baseline_times, baseline_err, label="Baseline translation error")
    if len(guided_pred):
        guided_err = np.linalg.norm(guided_pred[:, :3, 3] - guided_gt[:, :3, 3], axis=1)
        plt.scatter(guided_times, guided_err, s=18, label="IMU-guided translation error")
    plt.xlabel("time (s)")
    plt.ylabel("position error")
    plt.title("Aligned trajectory error over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_count_plot(output_path, baseline_count, guided_count):
    attention_ratio = (guided_count ** 2) / max(baseline_count ** 2, 1)
    plt.figure(figsize=(6, 4))
    labels = ["Baseline frames", "IMU-guided frames", "Attention proxy"]
    values = [baseline_count, guided_count, attention_ratio * baseline_count]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    plt.bar(labels, values, color=colors)
    plt.title("Sequence reduction proxy")
    plt.ylabel("count / proxy-scaled count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main(args):
    sequence = load_advio_iphone_sequence(args.advio_root)

    baseline_rows = load_frame_metadata(args.baseline_frames_csv)
    baseline_gt = interpolate_poses(
        np.asarray([timestamp for _, timestamp in baseline_rows], dtype=np.float64),
        sequence.ground_truth_timestamps,
        sequence.ground_truth_poses,
    )
    baseline_pred = load_predicted_pose_dir(args.baseline_pose_dir)
    aligned_baseline, gt_baseline_selected, baseline_times = align_predicted_to_gt(baseline_pred, baseline_rows, baseline_gt)

    guided_rows = load_frame_metadata(args.imu_guided_frames_csv)
    guided_gt = interpolate_poses(
        np.asarray([timestamp for _, timestamp in guided_rows], dtype=np.float64),
        sequence.ground_truth_timestamps,
        sequence.ground_truth_poses,
    )
    guided_pred = load_predicted_pose_dir(args.imu_guided_pose_dir)
    aligned_guided, gt_guided_selected, guided_times = align_predicted_to_gt(guided_pred, guided_rows, guided_gt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    make_trajectory_plot(output_dir / "trajectory_topdown.png", gt_baseline_selected, aligned_baseline, gt_guided_selected, aligned_guided)
    make_gyro_plot(output_dir / "gyro_keyframes.png", sequence, baseline_times, guided_times)
    make_error_plot(output_dir / "translation_error_over_time.png", baseline_times, aligned_baseline, gt_baseline_selected, guided_times, aligned_guided, gt_guided_selected)
    make_count_plot(output_dir / "frame_count_proxy.png", len(baseline_rows), len(guided_rows))

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ADVIO benchmark results.")
    parser.add_argument("--advio-root", required=True)
    parser.add_argument("--baseline-frames-csv", required=True)
    parser.add_argument("--baseline-pose-dir", required=True)
    parser.add_argument("--imu-guided-frames-csv", required=True)
    parser.add_argument("--imu-guided-pose-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    main(parser.parse_args())
