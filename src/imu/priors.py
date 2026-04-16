import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _first_available(row, candidates, default=None):
    for key in candidates:
        if key in row and row[key] != "":
            return row[key]
    return default


def _canonicalize_row(row):
    return {key.strip().lower(): value.strip() for key, value in row.items() if key is not None}


def _parse_timestamp(raw_value):
    value = float(raw_value)
    if value > 1e15:
        return value * 1e-9
    if value > 1e12:
        return value * 1e-6
    if value > 1e10:
        return value * 1e-3
    return value


def load_imu_csv(path):
    path = Path(path)
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = _canonicalize_row(raw_row)
            timestamp = _first_available(
                row,
                ["timestamp", "timestamp_s", "time", "t", "timestamp_ns", "timestamp_us", "timestamp_ms"],
            )
            if timestamp is None:
                raise ValueError("IMU CSV must contain a timestamp column.")

            item = {"timestamp": _parse_timestamp(timestamp)}

            quat_wxyz = [
                _first_available(row, ["qw", "quat_w", "q_w"]),
                _first_available(row, ["qx", "quat_x", "q_x"]),
                _first_available(row, ["qy", "quat_y", "q_y"]),
                _first_available(row, ["qz", "quat_z", "q_z"]),
            ]
            quat_xyzw = [
                _first_available(row, ["qx", "quat_x", "q_x"]),
                _first_available(row, ["qy", "quat_y", "q_y"]),
                _first_available(row, ["qz", "quat_z", "q_z"]),
                _first_available(row, ["qw", "quat_w", "q_w"]),
            ]

            if all(value is not None for value in quat_wxyz):
                item["quat_wxyz"] = np.asarray([float(value) for value in quat_wxyz], dtype=np.float64)
                item["quat_xyzw"] = np.asarray(
                    [item["quat_wxyz"][1], item["quat_wxyz"][2], item["quat_wxyz"][3], item["quat_wxyz"][0]],
                    dtype=np.float64,
                )
            elif all(value is not None for value in quat_xyzw):
                item["quat_xyzw"] = np.asarray([float(value) for value in quat_xyzw], dtype=np.float64)

            gyro = [
                _first_available(row, ["gyro_x", "gx", "wx"]),
                _first_available(row, ["gyro_y", "gy", "wy"]),
                _first_available(row, ["gyro_z", "gz", "wz"]),
            ]
            if all(value is not None for value in gyro):
                item["gyro"] = np.asarray([float(value) for value in gyro], dtype=np.float64)

            position = [
                _first_available(row, ["pos_x", "px", "x", "tx"]),
                _first_available(row, ["pos_y", "py", "y", "ty"]),
                _first_available(row, ["pos_z", "pz", "z", "tz"]),
            ]
            if all(value is not None for value in position):
                item["position"] = np.asarray([float(value) for value in position], dtype=np.float64)

            rows.append(item)

    if not rows:
        raise ValueError(f"No IMU samples found in {path}.")

    rows.sort(key=lambda item: item["timestamp"])
    return rows


def load_frame_timestamps(path):
    path = Path(path)
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = _canonicalize_row(raw_row)
            image_name = _first_available(row, ["image", "image_name", "filename", "file", "name"])
            timestamp = _first_available(
                row,
                ["timestamp", "timestamp_s", "time", "t", "timestamp_ns", "timestamp_us", "timestamp_ms"],
            )
            if image_name is None or timestamp is None:
                raise ValueError("Frame timestamp CSV must contain image and timestamp columns.")
            rows.append((image_name, _parse_timestamp(timestamp)))

    rows.sort(key=lambda item: item[0])
    return rows


def _integrate_gyro_orientation(imu_rows):
    timestamps = np.asarray([row["timestamp"] for row in imu_rows], dtype=np.float64)
    rotations = [Rotation.identity()]

    for index in range(1, len(imu_rows)):
        prev = imu_rows[index - 1]
        curr = imu_rows[index]
        dt = max(curr["timestamp"] - prev["timestamp"], 0.0)
        gyro = 0.5 * (prev.get("gyro", np.zeros(3)) + curr.get("gyro", np.zeros(3)))
        delta = Rotation.from_rotvec(gyro * dt)
        rotations.append(rotations[-1] * delta)

    return timestamps, Rotation.concatenate(rotations)


def build_imu_pose_sequence(imu_rows, invert_orientation=False):
    timestamps = np.asarray([row["timestamp"] for row in imu_rows], dtype=np.float64)

    if all("quat_xyzw" in row for row in imu_rows):
        rotations = Rotation.from_quat(np.asarray([row["quat_xyzw"] for row in imu_rows], dtype=np.float64))
    elif all("gyro" in row for row in imu_rows):
        timestamps, rotations = _integrate_gyro_orientation(imu_rows)
    else:
        raise ValueError("IMU CSV needs either quaternion columns or gyro columns for orientation priors.")

    if invert_orientation:
        rotations = rotations.inv()

    if all("position" in row for row in imu_rows):
        positions = np.asarray([row["position"] for row in imu_rows], dtype=np.float64)
    else:
        positions = np.zeros((len(imu_rows), 3), dtype=np.float64)

    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(imu_rows), axis=0)
    poses[:, :3, :3] = rotations.as_matrix()
    poses[:, :3, 3] = positions
    return timestamps, poses


def build_frame_pose_priors(frame_timestamps, imu_timestamps, imu_poses):
    if len(imu_timestamps) < 2:
        raise ValueError("Need at least two IMU samples to interpolate pose priors.")

    frame_times = np.asarray([timestamp for _, timestamp in frame_timestamps], dtype=np.float64)
    clamped_times = np.clip(frame_times, imu_timestamps[0], imu_timestamps[-1])

    imu_rotations = Rotation.from_matrix(imu_poses[:, :3, :3])
    slerp = Slerp(imu_timestamps, imu_rotations)
    interp_rotations = slerp(clamped_times).as_matrix()

    interp_positions = np.stack(
        [
            np.interp(clamped_times, imu_timestamps, imu_poses[:, 0, 3]),
            np.interp(clamped_times, imu_timestamps, imu_poses[:, 1, 3]),
            np.interp(clamped_times, imu_timestamps, imu_poses[:, 2, 3]),
        ],
        axis=1,
    )

    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(frame_timestamps), axis=0)
    poses[:, :3, :3] = interp_rotations
    poses[:, :3, 3] = interp_positions
    names = [name for name, _ in frame_timestamps]
    return names, poses


def save_pose_sequence_as_txt(image_names, poses, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_name, pose in zip(image_names, poses):
        target = output_dir / f"{Path(image_name).stem}.txt"
        np.savetxt(target, pose, fmt="%.8f")

