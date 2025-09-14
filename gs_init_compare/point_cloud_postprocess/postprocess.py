from pathlib import Path
import torch

from gs_init_compare.depth_prediction.utils.point_cloud_export import (
    export_point_cloud_to_ply,
)

from pointcloud_subsampling import subsample_pointcloud

from .config import OutlierRemovalMethod, PointCloudPostprocessConfig
from sklearn.neighbors import LocalOutlierFactor

import numpy as np


def lof_outlier_removal(pts: torch.Tensor, rgbs: torch.Tensor):
    k = 40
    rgbs_np = rgbs.cpu().numpy() * 2
    pts_np = pts.cpu().numpy()
    pts_and_rgb = np.concatenate([pts_np, rgbs_np], axis=1)

    clf = LocalOutlierFactor(n_neighbors=k, n_jobs=-1)
    pred_pts_only = clf.fit_predict(pts_np)
    pred_pts_and_rgbs = clf.fit_predict(pts_and_rgb)
    return pred_pts_only == -1
    return pred_pts_and_rgbs == -1


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
    points_to_cam_slices: list[tuple[int, int]],
    config: PointCloudPostprocessConfig,
    device: str,
):
    if config.outlier_removal != OutlierRemovalMethod.off:
        outliers = get_outlier_removal_func(config.outlier_removal)(pts, rgbs)

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

    if config.nyquist_subsample:
        pts, rgbs, _, merged, mergedrgb = subsample_pointcloud(
            pts.cpu().numpy(),
            rgbs.cpu().numpy(),
            intrinsic_matrices,
            proj_matrices,
            image_sizes,
            points_to_cam_slices,
            config.nyquist_subsample_factor,
        )
        export_point_cloud_to_ply(
            merged,
            mergedrgb,
            Path.cwd(),
            "merged",
        )

    return torch.tensor(pts).to(device), torch.tensor(rgbs).to(device)
