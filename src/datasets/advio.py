import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ADVIO_SEQUENCE_TAGS = {
    1: {"venue": "Mall", "indoor": True, "stairs": False, "escalator": True, "elevator": False, "people": "moderate", "vehicles": False},
    2: {"venue": "Mall", "indoor": True, "stairs": False, "escalator": False, "elevator": False, "people": "moderate", "vehicles": False},
    3: {"venue": "Mall", "indoor": True, "stairs": False, "escalator": False, "elevator": False, "people": "moderate", "vehicles": False},
    4: {"venue": "Mall", "indoor": True, "stairs": False, "escalator": True, "elevator": False, "people": "moderate", "vehicles": False},
    5: {"venue": "Mall", "indoor": True, "stairs": True, "escalator": False, "elevator": False, "people": "moderate", "vehicles": False},
    6: {"venue": "Mall", "indoor": True, "stairs": False, "escalator": False, "elevator": False, "people": "high", "vehicles": False},
    7: {"venue": "Mall", "indoor": True, "stairs": False, "escalator": False, "elevator": True, "people": "low", "vehicles": False},
    8: {"venue": "Mall", "indoor": True, "stairs": False, "escalator": True, "elevator": False, "people": "low", "vehicles": False},
    9: {"venue": "Mall", "indoor": True, "stairs": False, "escalator": True, "elevator": False, "people": "low", "vehicles": False},
    10: {"venue": "Mall", "indoor": True, "stairs": False, "escalator": False, "elevator": False, "people": "low", "vehicles": False},
    11: {"venue": "Metro", "indoor": True, "stairs": False, "escalator": False, "elevator": False, "people": "high", "vehicles": True},
    12: {"venue": "Metro", "indoor": True, "stairs": False, "escalator": False, "elevator": False, "people": "high", "vehicles": True},
    13: {"venue": "Office", "indoor": True, "stairs": True, "escalator": False, "elevator": False, "people": "low", "vehicles": False},
    14: {"venue": "Office", "indoor": True, "stairs": True, "escalator": False, "elevator": True, "people": "low", "vehicles": False},
    15: {"venue": "Office", "indoor": True, "stairs": False, "escalator": False, "elevator": False, "people": "none", "vehicles": False},
    16: {"venue": "Office", "indoor": True, "stairs": True, "escalator": False, "elevator": False, "people": "none", "vehicles": False},
    17: {"venue": "Office", "indoor": True, "stairs": True, "escalator": False, "elevator": False, "people": "none", "vehicles": False},
    18: {"venue": "Office", "indoor": True, "stairs": True, "escalator": False, "elevator": True, "people": "none", "vehicles": False},
    19: {"venue": "Office", "indoor": True, "stairs": True, "escalator": False, "elevator": False, "people": "none", "vehicles": False},
    20: {"venue": "Outdoor", "indoor": False, "stairs": False, "escalator": False, "elevator": False, "people": "low", "vehicles": True},
    21: {"venue": "Outdoor", "indoor": False, "stairs": False, "escalator": False, "elevator": False, "people": "low", "vehicles": True},
    22: {"venue": "Outdoor urban", "indoor": False, "stairs": False, "escalator": False, "elevator": False, "people": "high", "vehicles": True},
    23: {"venue": "Outdoor urban", "indoor": False, "stairs": False, "escalator": False, "elevator": False, "people": "high", "vehicles": True},
}


@dataclass
class ADVIOIPhoneSequence:
    sequence_id: int
    root: Path
    iphone_dir: Path
    ground_truth_dir: Path
    frame_timestamps: np.ndarray
    accelerometer: np.ndarray
    gyroscope: np.ndarray
    ground_truth_poses: np.ndarray
    ground_truth_timestamps: np.ndarray
    arkit_poses: np.ndarray | None
    arkit_timestamps: np.ndarray | None
    tags: dict


def _read_csv_rows(path):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            rows.append([item.strip() for item in row])
    return rows


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_rows(rows):
    normalized = []
    for row in rows:
        values = [_float_or_none(item) for item in row]
        if values[0] is None:
            continue
        normalized.append(values)
    return np.asarray(normalized, dtype=np.float64)


def _load_numeric_csv(path):
    rows = _read_csv_rows(path)
    return _normalize_rows(rows)


def _resolve_existing_path(*candidates):
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


def load_frame_timestamps_csv(path):
    data = _load_numeric_csv(path)
    return data[:, 0]


def _quat_candidates(values):
    if len(values) < 7:
        return None
    return [
        np.asarray(values[3:7], dtype=np.float64),  # qx,qy,qz,qw
        np.asarray(values[4:8], dtype=np.float64) if len(values) >= 8 else None,
    ]


def _rotation_from_pose_row(values):
    candidates = [candidate for candidate in _quat_candidates(values) if candidate is not None]
    for quat_xyzw in candidates:
        if np.all(np.isfinite(quat_xyzw)) and np.linalg.norm(quat_xyzw) > 0:
            try:
                return Rotation.from_quat(quat_xyzw / np.linalg.norm(quat_xyzw))
            except ValueError:
                continue
    raise ValueError("Could not parse quaternion from pose row.")


def load_pose_csv(path):
    data = _load_numeric_csv(path)
    timestamps = data[:, 0]
    poses = []
    for row in data:
        translation = np.asarray(row[1:4], dtype=np.float64)
        rotation = _rotation_from_pose_row(row).as_matrix()
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        poses.append(pose)
    return timestamps, np.asarray(poses)


def _extract_sequence_id(root):
    digits = "".join(character for character in root.name if character.isdigit())
    if not digits:
        raise ValueError(f"Could not infer ADVIO sequence id from {root.name}")
    return int(digits)


def load_advio_iphone_sequence(dataset_root):
    dataset_root = Path(dataset_root)
    iphone_dir = dataset_root / "iphone"
    ground_truth_dir = dataset_root / "ground-truth"

    sequence_id = _extract_sequence_id(dataset_root)
    frame_timestamps = load_frame_timestamps_csv(iphone_dir / "frames.csv")
    accelerometer = _load_numeric_csv(_resolve_existing_path(iphone_dir / "accelerometer.csv"))
    gyroscope = _load_numeric_csv(_resolve_existing_path(iphone_dir / "gyroscope.csv", iphone_dir / "gyro.csv"))
    ground_truth_timestamps, ground_truth_poses = load_pose_csv(
        _resolve_existing_path(ground_truth_dir / "poses.csv", ground_truth_dir / "pose.csv")
    )

    arkit_path = iphone_dir / "arkit.csv"
    if arkit_path.exists():
        arkit_timestamps, arkit_poses = load_pose_csv(arkit_path)
    else:
        arkit_timestamps, arkit_poses = None, None

    return ADVIOIPhoneSequence(
        sequence_id=sequence_id,
        root=dataset_root,
        iphone_dir=iphone_dir,
        ground_truth_dir=ground_truth_dir,
        frame_timestamps=frame_timestamps,
        accelerometer=accelerometer,
        gyroscope=gyroscope,
        ground_truth_poses=ground_truth_poses,
        ground_truth_timestamps=ground_truth_timestamps,
        arkit_poses=arkit_poses,
        arkit_timestamps=arkit_timestamps,
        tags=ADVIO_SEQUENCE_TAGS.get(sequence_id, {}),
    )


def align_timestamps(reference_timestamps, query_timestamps):
    reference_timestamps = np.asarray(reference_timestamps, dtype=np.float64)
    query_timestamps = np.asarray(query_timestamps, dtype=np.float64)
    indices = np.searchsorted(reference_timestamps, query_timestamps)
    indices = np.clip(indices, 0, len(reference_timestamps) - 1)
    left = np.clip(indices - 1, 0, len(reference_timestamps) - 1)
    choose_left = np.abs(reference_timestamps[left] - query_timestamps) < np.abs(reference_timestamps[indices] - query_timestamps)
    return np.where(choose_left, left, indices)


def interpolate_poses(sample_timestamps, pose_timestamps, poses):
    sample_timestamps = np.asarray(sample_timestamps, dtype=np.float64)
    pose_timestamps = np.asarray(pose_timestamps, dtype=np.float64)
    clamped = np.clip(sample_timestamps, pose_timestamps[0], pose_timestamps[-1])
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    slerp = Slerp(pose_timestamps, rotations)
    interp_rotations = slerp(clamped).as_matrix()
    interp_translation = np.stack(
        [
            np.interp(clamped, pose_timestamps, poses[:, 0, 3]),
            np.interp(clamped, pose_timestamps, poses[:, 1, 3]),
            np.interp(clamped, pose_timestamps, poses[:, 2, 3]),
        ],
        axis=1,
    )
    result = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(sample_timestamps), axis=0)
    result[:, :3, :3] = interp_rotations
    result[:, :3, 3] = interp_translation
    return result


def summarize_advio_sequence(sequence):
    return {
        "sequence_id": sequence.sequence_id,
        "num_frames": int(len(sequence.frame_timestamps)),
        "num_accelerometer": int(len(sequence.accelerometer)),
        "num_gyroscope": int(len(sequence.gyroscope)),
        "num_gt_poses": int(len(sequence.ground_truth_timestamps)),
        "tags": sequence.tags,
    }


def extract_video_frames(video_path, frame_timestamps, output_dir, image_prefix="frame", every_nth=1, limit=None, image_ext=".jpg"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = []
    frame_index = 0
    output_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index >= len(frame_timestamps):
            break

        if frame_index % every_nth == 0:
            image_name = f"{image_prefix}_{output_index:06d}{image_ext}"
            image_path = output_dir / image_name
            cv2.imwrite(str(image_path), frame)
            saved.append((image_name, float(frame_timestamps[frame_index]), frame_index))
            output_index += 1
            if limit is not None and output_index >= limit:
                break

        frame_index += 1

    capture.release()
    return saved
