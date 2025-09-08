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
    scene_scale: float,
    config: PointCloudPostprocessConfig,
    device: str
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
        if config.requires_descriptors():
            descriptors = descriptors[~outliers]
            patches = patches[~outliers]

    if config.octree_clustering:
        pts
    return pts.to(device), rgbs.to(device)

    # gc.collect()
    # # Pickle each item separately
    # items = {
    #     "points": pts.cpu(),
    #     "colors": rgbs.cpu(),
    #     "descriptors": descriptors.cpu(),
    #     "patches": patches.cpu(),
    # }

    # with Path("data.pkl").open("wb") as f:
    #     pickle.dump(items, f)

    # TODO: READ THIS FUCKING VIBE-CODED MESS
