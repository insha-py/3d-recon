import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from src.imu.keyframes import select_keyframes
from src.imu.priors import (
    build_frame_pose_priors,
    build_imu_pose_sequence,
    load_frame_timestamps,
    load_imu_csv,
    save_pose_sequence_as_txt,
)


def write_selection_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "selected_index"])
        writer.writerows(rows)


def main(args):
    images_dir = Path(args.images_dir)
    output_root = Path(args.output_root)

    imu_rows = load_imu_csv(args.imu_csv)
    frame_timestamps = load_frame_timestamps(args.frame_times_csv)

    imu_timestamps, imu_poses = build_imu_pose_sequence(
        imu_rows,
        invert_orientation=args.invert_orientation,
    )
    image_names, frame_poses = build_frame_pose_priors(frame_timestamps, imu_timestamps, imu_poses)

    selected_indices = select_keyframes(
        frame_poses,
        rotation_thresh_deg=args.rotation_thresh_deg,
        translation_thresh_m=args.translation_thresh_m,
        min_gap=args.min_gap,
    )

    selected_names = [image_names[index] for index in selected_indices]
    selected_poses = frame_poses[selected_indices]

    output_images = output_root / "images"
    output_pose = output_root / "pose"
    output_meta = output_root / "metadata"
    output_images.mkdir(parents=True, exist_ok=True)
    output_meta.mkdir(parents=True, exist_ok=True)

    for image_name in selected_names:
        source = images_dir / image_name
        if not source.exists():
            raise FileNotFoundError(f"Image listed in frame timestamps was not found: {source}")
        shutil.copy2(source, output_images / image_name)

    save_pose_sequence_as_txt(selected_names, selected_poses, output_pose)

    np.save(output_meta / "imu_frame_priors.npy", frame_poses)
    np.save(output_meta / "selected_pose_priors.npy", selected_poses)
    write_selection_csv(
        [(image_names[index], out_index) for out_index, index in enumerate(selected_indices)],
        output_meta / "selected_frames.csv",
    )

    reduction = 100.0 * (1.0 - (len(selected_indices) / max(len(image_names), 1)))
    print(f"Prepared IMU-guided dataset at: {output_root}")
    print(f"Original frames : {len(image_names)}")
    print(f"Selected frames : {len(selected_indices)}")
    print(f"Frame reduction : {reduction:.1f}%")
    print(f"Pose priors     : {output_pose}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a VGGT-ready dataset with IMU pose priors and motion-based keyframe selection."
    )
    parser.add_argument("--images-dir", required=True, help="Folder containing source images.")
    parser.add_argument("--imu-csv", required=True, help="CSV file with IMU timestamps and orientation data.")
    parser.add_argument(
        "--frame-times-csv",
        required=True,
        help="CSV with image_name,timestamp rows for aligning frames to IMU time.",
    )
    parser.add_argument("--output-root", required=True, help="Output dataset root with images/ and pose/.")
    parser.add_argument(
        "--invert-orientation",
        action="store_true",
        help="Invert IMU orientation if the sensor frame is world-to-device instead of device-to-world.",
    )
    parser.add_argument("--rotation-thresh-deg", type=float, default=4.0)
    parser.add_argument("--translation-thresh-m", type=float, default=0.05)
    parser.add_argument("--min-gap", type=int, default=1)
    main(parser.parse_args())
