import argparse
from pathlib import Path

from src.vggt.parse_colmap import parse_colmap_sparse
from src.geometry.pointcloud import create_pointcloud, save_pointcloud
from src.geometry.poses import save_poses
from src.visualization.visualize import visualize_reconstruction


def main(scan_id):

    sparse_dir = Path(f"data/scans/{scan_id}/sparse")

    xyz, rgb, poses, names = parse_colmap_sparse(sparse_dir)

    pcd = create_pointcloud(xyz, rgb)

    output_pcd = Path(f"outputs/pointclouds/{scan_id}.ply")
    output_pose = Path(f"outputs/poses/{scan_id}.npy")

    output_pcd.parent.mkdir(parents=True, exist_ok=True)
    output_pose.parent.mkdir(parents=True, exist_ok=True)

    save_pointcloud(pcd, output_pcd)
    save_poses(poses, output_pose)

    visualize_reconstruction(pcd, poses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", required=True)

    args = parser.parse_args()

    main(args.scan)
