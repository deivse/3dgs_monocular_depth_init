import logging
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from gs_init_compare.nerfbaselines_integration.method import gs_Parser as Parser
from gs_init_compare.monocular_depth_init.utils.point_cloud_export import (
    export_point_cloud_to_ply,
)

_LOGGER = logging.getLogger(__name__)


class LowDepthAlignmentConfidenceError(Exception):
    pass


def debug_export_point_clouds(
    imsize,
    cam2world,
    P,
    transform_c2w,
    sfm_points,
    pts_world,
    outliers_world,
    valid_indices,
    parser,
    image_name,
    downsample_factor,
    img,
    dir=Path("ply_export_debug"),
):
    camera_plane = torch.dstack(
        [
            torch.from_numpy(np.mgrid[0 : imsize[0], 0 : imsize[1]].T).to(
                cam2world.device
            ),
            torch.ones(imsize, device=cam2world.device).T,
        ]
    ).reshape(-1, 3)

    dir = Path(dir)
    dir.mkdir(exist_ok=True, parents=True)

    export_point_cloud_to_ply(
        transform_c2w(camera_plane).reshape(-1, 3).cpu().numpy(),
        img.reshape(-1, 3).cpu().numpy() if img is not None else None,
        dir,
        "camera_plane",
    )
    sfm_points_repro_world = P @ torch.vstack(
        [sfm_points.T, torch.ones(sfm_points.shape[0], device=cam2world.device)]
    )
    sfm_points_repro_world = sfm_points_repro_world / sfm_points_repro_world[2]
    sfm_pt_rgbs = parser.points_rgb[parser.point_indices[image_name]] / 255.0

    export_point_cloud_to_ply(
        transform_c2w(sfm_points_repro_world.T).reshape(-1, 3).cpu().numpy(),
        sfm_pt_rgbs,
        dir,
        "sfm_points_plane_proj",
    )
    export_point_cloud_to_ply(
        sfm_points.reshape(-1, 3).cpu().numpy(),
        sfm_pt_rgbs,
        dir,
        "sfm_points_world",
    )
    rgbs = None
    if img is not None:
        rgbs = img[::downsample_factor, ::downsample_factor]
        rgbs = rgbs.reshape(-1, 3).cpu().numpy()
        rgbs = rgbs[valid_indices.cpu().numpy()]

    export_point_cloud_to_ply(
        pts_world.reshape(-1, 3).cpu().numpy(),
        rgbs,
        dir,
        "my_points_world",
    )

    if outliers_world.shape[0] == 0:
        return

    red = np.array([1, 0, 0])
    export_point_cloud_to_ply(
        outliers_world.cpu().numpy(),
        np.tile(red, (outliers_world.shape[0], 1)),
        dir,
        "my_outliers_world",
    )


def get_depth_scalar(sfm_points, P, image_name, image_idx, imsize, depth, mask):
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
        if torch.sum(valid_sfm_pt_indices) < pts_camera.shape[1] / 4:
            raise LowDepthAlignmentConfidenceError(
                "Less than 1/4 of SFM points",
                f" ({torch.sum(valid_sfm_pt_indices).item()} / {sfm_points_camera.shape[1]})"
                f" reprojected into image bounds for image {image_name} ({image_idx})",
            )

        if mask is not None:
            # Set invalid points to 0 so we can index the mask with them
            # Will be filtered out later anyways
            sfm_points_camera[:, ~valid_sfm_pt_indices] = torch.zeros_like(
                sfm_points_camera[:, ~valid_sfm_pt_indices]
            )
            valid_sfm_pt_indices = torch.logical_and(
                valid_sfm_pt_indices, mask[sfm_points_camera[1], sfm_points_camera[0]]
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


def get_pts_from_depth(
    depth: torch.Tensor,
    mask: Optional[torch.Tensor],
    image_id: int,
    parser: Parser,
    cam2world: torch.Tensor,
    K: torch.Tensor,
    downsample_factor=10,
    outlier_factor=2.5,
    debug_point_cloud_export_dir: Optional[Path] = None,
    img_for_point_cloud_rgb=None,
):
    """
    Returns:
        pts_world: torch.Tensor on depth.device of shape [N, 3] where N is the number of points in the world space
        inlier_indices: torch.Tensor on CPU of shape [depth.shape[0] * depth.shape[1]] with True for inliers
    """
    depth = depth.float()
    imsize = depth.T.shape
    image_name = parser.image_names[image_id]
    w2c = torch.linalg.inv(cam2world)
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
        sfm_points, P, image_name, image_id, imsize, depth, mask
    )

    if depth_scalar is None:
        return None

    pts_camera: torch.Tensor = (
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

    valid_indices = torch.ones(
        pts_camera.shape[0], dtype=bool, device=pts_camera.device
    )
    if mask is not None:
        downsampled_mask = mask[::downsample_factor, ::downsample_factor].reshape(-1)
        valid_indices[~downsampled_mask] = 0

    inlier_indices = torch.abs(
        pts_camera[:, 2] - torch.mean(pts_camera[valid_indices, 2])
    ) < outlier_factor * torch.std(pts_camera[valid_indices, 2])
    outlier_indices = torch.logical_not(inlier_indices)

    # Only keep inliers that are not masked out
    inlier_indices = torch.logical_and(valid_indices, inlier_indices)
    outlier_indices = torch.logical_and(valid_indices, outlier_indices)

    inlier_ratio = float(
        torch.sum(inlier_indices).to(float) / torch.sum(valid_indices).to(float)
    )

    # Now valid indices filters out both masked out values and outliers
    valid_indices = inlier_indices

    pts_camera[:, 0] = (pts_camera[:, 0] + 0.5) * pts_camera[:, 2]
    pts_camera[:, 1] = (pts_camera[:, 1] + 0.5) * pts_camera[:, 2]

    pts_world_unfiltered = transform_camera_to_world_space(pts_camera)
    pts_world = pts_world_unfiltered[valid_indices]

    if debug_point_cloud_export_dir is not None:
        outliers_world = pts_world_unfiltered[outlier_indices]
        debug_export_point_clouds(
            imsize,
            cam2world,
            P,
            transform_camera_to_world_space,
            sfm_points,
            pts_world,
            outliers_world,
            valid_indices,
            parser,
            image_name,
            downsample_factor,
            img_for_point_cloud_rgb,
            debug_point_cloud_export_dir,
        )

    return pts_world.reshape([-1, 3]).float(), valid_indices.cpu(), inlier_ratio
