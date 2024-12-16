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
):
    """Saves point cloud to a .ply file."""
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    if rgbs is not None:
        pcd.colors = open3d.utility.Vector3dVector(rgbs)
    open3d.io.write_point_cloud(
        str(output_dir / f"{depth_pts_filename}.ply"), pcd, write_ascii=False
    )
    logging.info(f"Saved point cloud to {output_dir / depth_pts_filename}.ply")
