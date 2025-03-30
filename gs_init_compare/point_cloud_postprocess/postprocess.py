import numpy as np
import torch
import open3d as o3d

from .config import PointCloudPostprocessConfig


def postprocess_point_cloud(
    pts: torch.Tensor,
    rgbs: torch.Tensor,
    scene_scale: float,
    config: PointCloudPostprocessConfig,
):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
    point_cloud.colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
    if config.voxel_subsample:
        voxel_size = scene_scale * config.voxel_size_wrt_scene_extent
        point_cloud = point_cloud.voxel_down_sample(voxel_size)

    if config.outlier_removal:
        point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)

    pts = torch.tensor(
        np.asarray(point_cloud.points),
        device=pts.device,
        dtype=pts.dtype,
    )
    rgbs = torch.tensor(
        np.asarray(point_cloud.colors),
        device=rgbs.device,
        dtype=rgbs.dtype,
    )
    return pts, rgbs
