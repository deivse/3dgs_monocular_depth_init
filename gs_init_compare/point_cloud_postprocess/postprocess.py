from pathlib import Path
import torch

from gs_init_compare.utils.point_cloud_export import (
    export_point_cloud_to_ply,
)

from pointcloud_subsampling import subsample_pointcloud

from .config import OutlierRemovalMethod, PointCloudPostprocessConfig
from sklearn.neighbors import LocalOutlierFactor

import numpy as np


def lof_outlier_removal(pts: torch.Tensor, config: PointCloudPostprocessConfig):
    pts_np = pts.cpu().numpy()

    # TODO: what about scanning through possible K values as the paper suggests? https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf
    clf = LocalOutlierFactor(n_neighbors=config.lof_num_neighbors, n_jobs=-1)
    pred_pts_only = clf.fit_predict(pts_np)
    return pred_pts_only == -1


def get_outlier_removal_func(method: OutlierRemovalMethod):
    if method == OutlierRemovalMethod.lof:
        return lof_outlier_removal
    else:
        raise ValueError(f"Unknown outlier removal method: {method}")


def postprocess_point_cloud(
    pts: torch.Tensor,
    rgbs: torch.Tensor,
    intrinsic_matrices: list[np.ndarray],
    proj_matrices: list[np.ndarray],
    image_sizes: np.ndarray,
    config: PointCloudPostprocessConfig,
    device: str,
):
    if config.subsample:
        pts, rgbs, _, merged, mergedrgb = subsample_pointcloud(
            pts.cpu().numpy(),
            rgbs.cpu().numpy(),
            intrinsic_matrices,
            proj_matrices,
            image_sizes,
            params=config.subsample_params,
        )
        export_point_cloud_to_ply(
            merged,
            mergedrgb,
            Path.cwd(),
            "merged",
        )
        pts, rgbs = torch.from_numpy(pts).to(device), torch.from_numpy(rgbs).to(device)

    # Perform after subsampling because more points may be considered outliers after their "duplicates" are removed
    if config.outlier_removal != OutlierRemovalMethod.off:
        outliers = get_outlier_removal_func(config.outlier_removal)(pts, config)

        export_point_cloud_to_ply(
            pts[~outliers].cpu().numpy(),
            rgbs[~outliers].cpu().numpy(),
            Path.cwd(),
            f"filtered_{config.outlier_removal.value}",
        )
        export_point_cloud_to_ply(
            pts[outliers].cpu().numpy(),
            rgbs[outliers].cpu().numpy(),
            Path.cwd(),
            f"outliers_{config.outlier_removal.value}",
        )

        pts = pts[~outliers]
        rgbs = rgbs[~outliers]

    return pts, rgbs
