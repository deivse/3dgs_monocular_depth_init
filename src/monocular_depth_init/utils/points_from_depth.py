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
    device = sfm_points.device

    sfm_points_camera = P @ torch.vstack(
        [sfm_points.T, torch.ones(sfm_points.shape[0], device=device)]
    )
    sfm_points_depth = sfm_points_camera[2]
    sfm_points_camera = sfm_points_camera[:2] / sfm_points_camera[2]

    def get_valid_sfm_pts(pts_camera, pts_camera_depth):
        valid_sfm_pt_indices = torch.logical_and(
            torch.logical_and(pts_camera[0] >= 0, pts_camera[0] < imsize[0]),
            torch.logical_and(pts_camera[1] >= 0, pts_camera[1] < imsize[1]),
        )
        valid_sfm_pt_indices = torch.logical_and(
            valid_sfm_pt_indices, pts_camera_depth >= 0
        )
        if torch.sum(valid_sfm_pt_indices) < pts_camera.shape[1] * 3.0 / 4.0:
            _LOGGER.warning(
                "Only %s/%s SFM points reprojected into image bounds for image %s (%s)",
                torch.sum(valid_sfm_pt_indices),
                sfm_points_camera.shape[1],
                image_name,
                image_idx,
            )
        return sfm_points_camera[:, valid_sfm_pt_indices], sfm_points_depth[
            valid_sfm_pt_indices
        ]

    sfm_points_camera = torch.round(sfm_points_camera).to(int)
    sfm_points_camera, sfm_points_depth = get_valid_sfm_pts(
        sfm_points_camera, sfm_points_depth
    )
    predicted_depth: torch.Tensor = depth[sfm_points_camera[1], sfm_points_camera[0]]
    d = torch.vstack(
        [
            predicted_depth.reshape(-1),
            torch.ones(predicted_depth.shape[0], device=device),
        ]
    )

    # Equations 2-5 in "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
    # https://arxiv.org/pdf/1907.01341
    h = torch.sum(d * sfm_points_depth, axis=1) / torch.linalg.norm(d) ** 2
    return h[0], h[1], sfm_points_camera


@dataclass
class DebugPlotConfig:
    show_sfm_points: bool = True
    show_outliers: bool = True
    show_camera_plane: bool = True
    sfm_pts_downsample_factor: int = 10
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
            torch.abs(pts_camera[:, 2] - torch.mean(pts_camera[:, 2]))
            >= outlier_factor * torch.std(pts_camera[:, 2])
        ]

        outliers[:, 0] = (outliers[:, 0] + 0.5) * outliers[:, 2]
        outliers[:, 1] = (outliers[:, 1] + 0.5) * outliers[:, 2]

        outliers = transform_camera_to_world_space(outliers)

        camera_plane_xyz = transform_camera_to_world_space(
            torch.dstack(
                [
                    torch.from_numpy(np.mgrid[0 : imsize[0], 0 : imsize[1]].T).to(
                        depth.device
                    ),
                    depth_scalar * torch.ones(depth.shape, device=depth.device),
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
            plot3d(sfm_points[:: self.sfm_pts_downsample_factor, :].cpu(), "g", ax)
        if self.show_outliers:
            plot3d(outliers.cpu(), "r", ax)
        if self.show_camera_plane:
            plot3d(camera_plane_xyz.cpu(), "b", ax)
        plot3d(pts_world.cpu(), "k", ax)
        plt.show(block=True)


def get_pts_from_depth(
    depth: torch.Tensor,
    image_name: str,
    image_idx: int,
    parser: Parser,
    downsample_factor=10,
    outlier_factor=2.5,
    debug_plot_conf: Optional[DebugPlotConfig] = None,
):
    """
    Returns:
        pts_world: torch.Tensor on depth.device of shape [N, 3] where N is the number of points in the world space
        inlier_indices: torch.Tensor on CPU of shape [depth.shape[0] * depth.shape[1]] with True for inliers
    """
    depth = depth.float()
    cam2world = torch.from_numpy(parser.cam_to_worlds[image_idx])
    camera_id = parser.camera_ids[image_idx]
    imsize = parser.imsize_dict[camera_id]
    w2c = torch.linalg.inv(cam2world)
    K = torch.from_numpy(parser.Ks_dict[camera_id])
    R = w2c[:3, :3]
    C = -R.T @ w2c[:3, 3]
    P = K @ R @ torch.hstack([torch.eye(3), -C[:, None]])

    sfm_points = (
        torch.from_numpy(parser.points[parser.point_indices[image_name]])
        .to(depth.device)
        .float()
    )

    cam2world = cam2world.to(depth.device).float()
    P = P.to(depth.device).float()
    K = K.to(depth.device).float()

    def transform_camera_to_world_space(camera_homo: torch.Tensor) -> torch.Tensor:
        dense_world = torch.linalg.inv(K) @ camera_homo.reshape((-1, 3)).T
        dense_world = (
            cam2world
            @ torch.vstack(
                [dense_world, torch.ones(dense_world.shape[1], device=cam2world.device)]
            )
        )[:3].T
        return dense_world

    depth_scalar, depth_shift, sfm_points_camera_homo = get_depth_scalar(
        sfm_points, P, image_name, image_idx, imsize, depth
    )

    if depth_scalar is None:
        return None

    pts_camera = (
        torch.dstack(
            [
                torch.from_numpy(np.mgrid[0 : imsize[0], 0 : imsize[1]].T).to(
                    depth.device
                ),
                depth_scalar * depth + depth_shift,
            ],
        )[::downsample_factor, ::downsample_factor, :]
        .reshape(-1, 3)
        .to(depth.device)
    )

    inlier_indices = torch.abs(
        pts_camera[:, 2] - torch.mean(pts_camera[:, 2])
    ) < outlier_factor * torch.std(pts_camera[:, 2])
    print(
        "Inlier depth ratio:",
        float(torch.sum(inlier_indices).to(float) / pts_camera.shape[0]),
    )
    pts_camera = pts_camera[inlier_indices]

    pts_camera[:, 0] = (pts_camera[:, 0] + 0.5) * pts_camera[:, 2]
    pts_camera[:, 1] = (pts_camera[:, 1] + 0.5) * pts_camera[:, 2]

    pts_world = transform_camera_to_world_space(pts_camera)

    if debug_plot_conf is not None:
        debug_plot_conf.plot(
            sfm_points,
            pts_camera,
            pts_world,
            outlier_factor,
            imsize,
            depth_scalar,
            depth,
            transform_camera_to_world_space,
        )

    return pts_world.reshape([-1, 3]).float(), inlier_indices.cpu()
