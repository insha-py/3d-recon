import struct
import collections
import numpy as np
from pathlib import Path


Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)

Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name"]
)

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error"]
)


def read_next_bytes(fid, num_bytes, format_sequence):
    if format_sequence and format_sequence[0] not in "@=<>!":
        format_sequence = "<" + format_sequence
    expected_bytes = struct.calcsize(format_sequence)
    if num_bytes != expected_bytes:
        num_bytes = expected_bytes
    data = fid.read(num_bytes)
    return struct.unpack(format_sequence, data)


# ============================
# Cameras
# ============================

def read_cameras_binary(path):

    cameras = {}

    with open(path, "rb") as fid:

        num_cameras = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_cameras):

            camera_properties = read_next_bytes(fid, 24, "iiQQ")

            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]

            # SIMPLE_PINHOLE → 3 params
            num_params = 3

            params = read_next_bytes(fid, num_params * 8, "ddd")

            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_id,
                width=width,
                height=height,
                params=np.array(params)
            )

    return cameras


# ============================
# Images
# ============================

def read_images_binary(path):

    images = {}

    with open(path, "rb") as fid:

        num_images = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_images):

            binary_tuple = read_next_bytes(fid, 64, "idddddddi")

            image_id = binary_tuple[0]

            qvec = np.array(binary_tuple[1:5])

            tvec = np.array(binary_tuple[5:8])

            camera_id = binary_tuple[8]

            # read image name
            name = ""

            while True:
                char = fid.read(1).decode("utf-8")

                if char == "\x00":
                    break

                name += char

            # skip 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]

            fid.seek(num_points2D * 24, 1)

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name
            )

    return images


# ============================
# Points3D
# ============================

def read_points3d_binary(path):

    points3D = {}

    with open(path, "rb") as fid:

        num_points = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_points):

            binary_tuple = read_next_bytes(fid, 43, "QdddBBBd")

            point_id = binary_tuple[0]

            xyz = np.array(binary_tuple[1:4])

            rgb = np.array(binary_tuple[4:7])

            error = binary_tuple[7]

            track_length = read_next_bytes(fid, 8, "Q")[0]

            fid.seek(track_length * 8, 1)

            points3D[point_id] = Point3D(
                id=point_id,
                xyz=xyz,
                rgb=rgb,
                error=error
            )

    return points3D


# ============================
# Main read function
# ============================

def read_model(path):

    path = Path(path)

    cameras = read_cameras_binary(path / "cameras.bin")

    images = read_images_binary(path / "images.bin")

    points3D = read_points3d_binary(path / "points3D.bin")

    return cameras, images, points3D
