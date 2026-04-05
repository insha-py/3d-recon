import os
import subprocess
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_ROOT = "data"


# -----------------------
# Read COLMAP points3D.bin
# -----------------------
def read_points3d_binary(path):
    points = []

    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_points):
            f.read(8)  # point id
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            f.read(8)  # error

            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(track_length * 8)

            points.append((*xyz, *rgb))

    return points


# -----------------------
# Save PLY
# -----------------------
def save_ply(points, output_path):
    with open(output_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {int(p[3])} {int(p[4])} {int(p[5])}\n")


# -----------------------
# Visualization
# -----------------------
def visualize(points, title="Point Cloud"):
    # subsample for speed
    pts = points[::50]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, s=1)

    ax.set_title(title)
    plt.show()


# -----------------------
# Process all scans
# -----------------------
for scan in os.listdir(DATA_ROOT):
    scene_dir = os.path.join(DATA_ROOT, scan)

    if not os.path.isdir(scene_dir):
        continue

    image_dir = os.path.join(scene_dir, "images")
    if not os.path.exists(image_dir):
        print(f"Skipping {scan}, no images folder")
        continue

    print(f"\n🚀 Processing {scan}")

    # -----------------------
    # Run VGGT COLMAP export
    # -----------------------
    subprocess.run([
        "python", "demo_colmap.py",
        f"--scene_dir={scene_dir}",
        "--max_query_pts=2048",
        "--query_frame_num=5"
    ])

    # -----------------------
    # Convert to PLY
    # -----------------------
    points_bin = os.path.join(scene_dir, "sparse", "points3D.bin")

    if not os.path.exists(points_bin):
        print(f"❌ No points3D.bin found for {scan}")
        continue

    points = read_points3d_binary(points_bin)

    ply_path = os.path.join(scene_dir, "pointcloud.ply")
    save_ply(points, ply_path)

    print(f"✅ Saved PLY: {ply_path}")

    # -----------------------
    # VISUALIZE
    # -----------------------
    print(f"🎥 Visualizing {scan}...")
    visualize(points, title=scan)

print("\n🎉 Done processing all scans!")