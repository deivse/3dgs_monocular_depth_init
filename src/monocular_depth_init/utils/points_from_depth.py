from dataclasses import dataclass
import logging
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import torch

from datasets.colmap import Parser
from monocular_depth_init.utils.plot3d import plot3d

_LOGGER = logging.getLogger(__name__)


def get_depth_scalar(sfm_points, P, image_name, image_idx, imsize, depth):
    sfm_points_camera = P @ np.vstack([sfm_points.T, np.ones(sfm_points.shape[0])])
    sfm_points_depth = sfm_points_camera[2]
    sfm_points_camera = sfm_points_camera[:2] / sfm_points_camera[2]

    def get_valid_sfm_pts(pts_camera, pts_camera_depth):
        valid_sfm_pt_indices = np.logical_and(
            np.logical_and(pts_camera[0] >= 0, pts_camera[0] < imsize[0]),
            np.logical_and(pts_camera[1] >= 0, pts_camera[1] < imsize[1]),
        )
        valid_sfm_pt_indices = np.logical_and(
            valid_sfm_pt_indices, pts_camera_depth >= 0
        )
        if np.sum(valid_sfm_pt_indices) < pts_camera.shape[1] * 3.0 / 4.0:
            _LOGGER.warning(
                "Only %s/%s SFM points reprojected into image bounds for image %s (%s)",
                np.sum(valid_sfm_pt_indices),
                sfm_points_camera.shape[1],
                image_name,
                image_idx,
            )
        return sfm_points_camera[:, valid_sfm_pt_indices], sfm_points_depth[
            valid_sfm_pt_indices
        ]

    sfm_points_camera = np.round(sfm_points_camera).astype(int)
    sfm_points_camera, sfm_points_depth = get_valid_sfm_pts(
        sfm_points_camera, sfm_points_depth
    )
    predicted_depth = depth[sfm_points_camera[1], sfm_points_camera[0]]
    d = np.vstack([predicted_depth.reshape(-1), np.ones(predicted_depth.shape[0])])

    # Equations 2-5 in "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
    # https://arxiv.org/pdf/1907.01341
    h = np.sum(d * sfm_points_depth, axis=1) / np.linalg.norm(d) ** 2
    return h[0], h[1], sfm_points_camera


@dataclass
class DebugPlotConfig:
    show_sfm_points: bool
    show_outliers: bool
    show_camera_plane: bool
    sfm_pts_downsample_factor: int
    camera_plane_downsample_factor: int = 40

    def plot(
        self,
        sfm_points,
        pts_camera,
        pts_world,
        outlier_factor,
        imsize,
        depth_scalar,
        depth,
        transform_camera_to_world_space,
    ):
        outliers = pts_camera[
            np.abs(pts_camera[:, 2] - np.mean(pts_camera[:, 2]))
            >= outlier_factor * np.std(pts_camera[:, 2])
        ]

        outliers[:, 0] = (outliers[:, 0] + 0.5) * outliers[:, 2]
        outliers[:, 1] = (outliers[:, 1] + 0.5) * outliers[:, 2]

        outliers = transform_camera_to_world_space(outliers)

        camera_plane_xyz = transform_camera_to_world_space(
            np.dstack(
                [
                    np.mgrid[0 : imsize[0], 0 : imsize[1]].T,
                    depth_scalar * np.ones(depth.shape),
                ]
            )[
                :: self.camera_plane_downsample_factor,
                :: self.camera_plane_downsample_factor,
                :,
            ],
        )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if self.show_sfm_points:
            plot3d(sfm_points[:: self.sfm_pts_downsample_factor, :], "g", ax)
        if self.show_outliers:
            plot3d(outliers, "r", ax)
        if self.show_camera_plane:
            plot3d(camera_plane_xyz, "b", ax)
        plot3d(pts_world, "k", ax)
        plt.show(block=True)


def get_pts_from_depth(
    depth: np.ndarray,
    image_name: str,
    image_idx: int,
    parser: Parser,
    downsample_factor=10,
    outlier_factor=2.5,
    debug_plot_conf: Optional[DebugPlotConfig] = None,
):
    cam2world = parser.cam_to_worlds[image_idx]
    camera_id = parser.camera_ids[image_idx]
    K = parser.Ks_dict[camera_id]
    imsize = parser.imsize_dict[camera_id]
    w2c = np.linalg.inv(cam2world)
    R = w2c[:3, :3]
    C = -R.T @ w2c[:3, 3]
    P = K @ R @ np.hstack([np.eye(3), -C[:, None]])

    sfm_points = parser.points[parser.point_indices[image_name]]

    def transform_camera_to_world_space(camera_homo) -> np.ndarray:
        dense_world = np.linalg.inv(K) @ camera_homo.reshape((-1, 3)).T
        dense_world = (
            cam2world @ np.vstack([dense_world, np.ones(dense_world.shape[1])])
        )[:3].T
        return dense_world

    depth_scalar, depth_shift, sfm_points_camera_homo = get_depth_scalar(
        sfm_points, P, image_name, image_idx, imsize, depth
    )

    if depth_scalar is None:
        return None

    pts_camera = np.dstack(
        [
            np.mgrid[0 : imsize[0], 0 : imsize[1]].T,
            depth_scalar * depth + depth_shift,
        ]
    )[::downsample_factor, ::downsample_factor, :].reshape(-1, 3)

    inlier_indices = np.abs(
        pts_camera[:, 2] - np.mean(pts_camera[:, 2])
    ) < outlier_factor * np.std(pts_camera[:, 2])
    print(
        "Inlier depth ratio:",
        np.sum(inlier_indices).astype(float) / pts_camera.shape[0],
    )
    pts_camera = pts_camera[inlier_indices]

    pts_camera[:, 0] = (pts_camera[:, 0] + 0.5) * pts_camera[:, 2]
    pts_camera[:, 1] = (pts_camera[:, 1] + 0.5) * pts_camera[:, 2]

    pts_world = transform_camera_to_world_space(pts_camera)

    return torch.from_numpy(pts_world.reshape([-1, 3])), inlier_indices
