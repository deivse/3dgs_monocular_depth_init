import logging
from pathlib import Path
from typing import Optional

import numpy as np
import open3d


def export_point_cloud_to_ply(
    pts: np.ndarray,
    rgbs: Optional[np.ndarray],
    output_dir: Path,
    depth_pts_filename: str,
    outlier_std_dev: Optional[float] = None,
):
    """Saves point cloud to a .ply file."""
    if outlier_std_dev is not None:
        # Remove outliers
        mean = np.mean(pts, axis=0)
        std_dev = np.std(pts, axis=0)
        mask = np.all(np.abs((pts - mean) / std_dev) < outlier_std_dev, axis=1)
        pts = pts[mask]
        if rgbs is not None:
            rgbs = rgbs[mask]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    if rgbs is not None:
        pcd.colors = open3d.utility.Vector3dVector(rgbs)
    open3d.io.write_point_cloud(
        str(output_dir / f"{depth_pts_filename}.ply"), pcd, write_ascii=False
    )
    logging.info(f"Saved point cloud to {output_dir / depth_pts_filename}.ply")
