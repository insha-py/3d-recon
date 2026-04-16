import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from src.datasets.advio import extract_video_frames, load_advio_iphone_sequence, summarize_advio_sequence
from src.imu.keyframes import select_keyframes
from src.imu.priors import build_frame_pose_priors, build_imu_pose_sequence, save_pose_sequence_as_txt


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def prepare_imu_rows(accelerometer, gyroscope):
    gyro_map = {float(row[0]): np.asarray(row[1:4], dtype=np.float64) for row in gyroscope}
    rows = []
    for acc_row in accelerometer:
        timestamp = float(acc_row[0])
        row = {"timestamp": timestamp}
        if timestamp in gyro_map:
            row["gyro"] = gyro_map[timestamp]
        else:
            nearest = np.argmin(np.abs(gyroscope[:, 0] - timestamp))
            row["gyro"] = np.asarray(gyroscope[nearest, 1:4], dtype=np.float64)
        row["accel"] = np.asarray(acc_row[1:4], dtype=np.float64)
        rows.append(row)
    return rows


def prepare_vggt_images(args, sequence, output_root):
    extraction_root = output_root / "baseline"
    extracted_rows = extract_video_frames(
        sequence.iphone_dir / "frames.mov",
        sequence.frame_timestamps,
        extraction_root / "images",
        image_prefix="frame",
        every_nth=args.every_nth_frame,
        limit=args.max_frames,
        image_ext=args.image_ext,
    )

    baseline_names = [name for name, _, _ in extracted_rows]
    write_csv(extraction_root / "metadata" / "frame_times.csv", ["image_name", "timestamp", "source_frame_index"], extracted_rows)
    return extraction_root, baseline_names, extracted_rows


def build_imu_guided_subset(sequence, extracted_rows, output_root, args):
    imu_rows = prepare_imu_rows(sequence.accelerometer, sequence.gyroscope)
    imu_timestamps, imu_poses = build_imu_pose_sequence(imu_rows)
    frame_timestamps = [(name, timestamp) for name, timestamp, _ in extracted_rows]
    frame_names, frame_priors = build_frame_pose_priors(frame_timestamps, imu_timestamps, imu_poses)

    selected_indices = select_keyframes(
        frame_priors,
        rotation_thresh_deg=args.rotation_thresh_deg,
        translation_thresh_m=args.translation_thresh_m,
        min_gap=args.min_gap,
    )

    guided_root = output_root / "imu_guided"
    guided_images = guided_root / "images"
    guided_metadata = guided_root / "metadata"
    guided_images.mkdir(parents=True, exist_ok=True)
    guided_metadata.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    selected_names = []
    selected_priors = frame_priors[selected_indices]
    for output_index, source_index in enumerate(selected_indices):
        image_name, timestamp, original_frame_index = extracted_rows[source_index]
        shutil.copy2(output_root / "baseline" / "images" / image_name, guided_images / image_name)
        selected_rows.append((image_name, timestamp, original_frame_index, source_index))
        selected_names.append(image_name)

    save_pose_sequence_as_txt(selected_names, selected_priors, guided_root / "pose")
    np.save(guided_metadata / "imu_frame_priors.npy", frame_priors)
    np.save(guided_metadata / "selected_pose_priors.npy", selected_priors)
    write_csv(
        guided_metadata / "selected_frames.csv",
        ["image_name", "timestamp", "source_frame_index", "baseline_frame_index"],
        selected_rows,
    )

    return guided_root, selected_rows, frame_priors, selected_priors


def main(args):
    sequence = load_advio_iphone_sequence(args.advio_root)
    output_root = Path(args.output_root)

    baseline_root, baseline_names, extracted_rows = prepare_vggt_images(args, sequence, output_root)
    guided_root, selected_rows, frame_priors, selected_priors = build_imu_guided_subset(sequence, extracted_rows, output_root, args)

    summary = summarize_advio_sequence(sequence)
    summary["baseline_frames"] = len(baseline_names)
    summary["imu_guided_frames"] = len(selected_rows)
    summary["frame_reduction_percent"] = 100.0 * (1.0 - len(selected_rows) / max(len(baseline_names), 1))
    summary["rotation_threshold_deg"] = args.rotation_thresh_deg
    summary["translation_threshold_m"] = args.translation_thresh_m
    summary["min_gap"] = args.min_gap
    summary["every_nth_frame"] = args.every_nth_frame
    summary["max_frames"] = args.max_frames

    write_json(output_root / "summary.json", summary)

    print(f"Prepared ADVIO sequence {sequence.sequence_id} at {output_root}")
    print(f"Baseline frames   : {len(baseline_names)}")
    print(f"IMU-guided frames : {len(selected_rows)}")
    print(f"Frame reduction   : {summary['frame_reduction_percent']:.1f}%")
    print(f"Baseline images   : {baseline_root / 'images'}")
    print(f"IMU-guided images : {guided_root / 'images'}")
    print(f"IMU pose priors   : {guided_root / 'pose'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ADVIO iPhone data for baseline and IMU-guided VGGT runs.")
    parser.add_argument("--advio-root", required=True, help="Path to one ADVIO sequence root, e.g. data/advio-01.")
    parser.add_argument("--output-root", required=True, help="Output root for extracted images and metadata.")
    parser.add_argument("--every-nth-frame", type=int, default=1, help="Temporal frame stride when extracting the video.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on extracted frames.")
    parser.add_argument("--image-ext", default=".jpg", help="Extension for extracted images.")
    parser.add_argument("--rotation-thresh-deg", type=float, default=4.0)
    parser.add_argument("--translation-thresh-m", type=float, default=0.0)
    parser.add_argument("--min-gap", type=int, default=1)
    main(parser.parse_args())
