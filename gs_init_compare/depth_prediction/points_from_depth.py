import logging
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from matplotlib import pyplot as plt

from gs_init_compare.config import Config
from gs_init_compare.datasets.colmap import Parser
from gs_init_compare.depth_alignment.interface import DepthAlignmentResult
from gs_init_compare.depth_alignment.pipeline import DepthAlignmentPipeline
from gs_init_compare.depth_prediction.predictors.depth_predictor_interface import (
    PredictedDepth,
)
from gs_init_compare.depth_subsampling.adaptive_subsampling import (
    AdaptiveDepthSubsampler,
)
from gs_init_compare.depth_subsampling.static_subsampler import StaticDepthSubsampler
from gs_init_compare.nerfbaselines_integration.method import (
    gs_Parser as NerfbaselinesParser,
)
from gs_init_compare.utils.point_cloud_export import (
    export_point_cloud_to_ply,
)
from gs_init_compare.types import InputImage

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
    parser,
    image_name,
    downsample_mask,
    rgb_image,
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
        rgb_image.reshape(-1, 3).cpu().numpy() if rgb_image is not None else None,
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
    if rgb_image is not None:
        rgbs = rgb_image.view(-1, 3)[downsample_mask]
        rgbs = rgbs.cpu().numpy()

    export_point_cloud_to_ply(
        pts_world.reshape(-1, 3).cpu().numpy(),
        rgbs,
        dir,
        "my_points_world",
    )


def get_valid_sfm_pts(
    sfm_pts_camera, sfm_pts_camera_depth, mask, imsize
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_sfm_pt_indices = torch.logical_and(
        torch.logical_and(sfm_pts_camera[0] >= 0, sfm_pts_camera[0] < imsize[0]),
        torch.logical_and(sfm_pts_camera[1] >= 0, sfm_pts_camera[1] < imsize[1]),
    )
    valid_sfm_pt_indices = torch.logical_and(
        valid_sfm_pt_indices, sfm_pts_camera_depth >= 0
    )
    print(
        f"Num invalid reprojected SfM points: {sfm_pts_camera.shape[1] - torch.sum(valid_sfm_pt_indices)} out of {sfm_pts_camera.shape[1]}"
    )
    if torch.sum(valid_sfm_pt_indices) < sfm_pts_camera.shape[1] / 4:
        raise LowDepthAlignmentConfidenceError(
            "Less than 1/4 of SFM points",
            f" ({torch.sum(valid_sfm_pt_indices).item()} / {sfm_pts_camera.shape[1]})"
            f" reprojected into image bounds.",
        )

    # Set invalid points to 0 so we can index the mask with them
    # Will be filtered out later anyways
    sfm_pts_camera[:, ~valid_sfm_pt_indices] = torch.zeros_like(
        sfm_pts_camera[:, ~valid_sfm_pt_indices]
    )
    valid_sfm_pt_indices = torch.logical_and(
        valid_sfm_pt_indices, mask[sfm_pts_camera[1], sfm_pts_camera[0]]
    )

    return sfm_pts_camera[:, valid_sfm_pt_indices], sfm_pts_camera_depth[
        valid_sfm_pt_indices
    ]


def align_depth(
    image: torch.Tensor,
    config: Config,
    sfm_points: torch.Tensor,
    P: torch.Tensor,
    imsize: torch.Tensor,
    predicted_depth: PredictedDepth,
    debug_export_dir: Path | None,
) -> DepthAlignmentResult:
    device = sfm_points.device

    sfm_points_camera = P @ torch.vstack(
        [sfm_points.T, torch.ones(sfm_points.shape[0], device=device)]
    )
    sfm_points_depth = sfm_points_camera[2]
    sfm_points_camera = sfm_points_camera[:2] / sfm_points_camera[2]

    sfm_points_camera = torch.round(sfm_points_camera).to(int)
    sfm_points_camera, sfm_points_depth = get_valid_sfm_pts(
        sfm_points_camera, sfm_points_depth, predicted_depth.mask, imsize
    )
    if debug_export_dir is not None:
        debug_export_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 10))
        plt.scatter(
            sfm_points_camera[0].cpu(),
            sfm_points_camera[1].cpu(),
            s=1,
            c="red",
        )
        plt.imshow(image.clone().cpu())
        plt.axis("off")
        plt.savefig(
            debug_export_dir / "sfm_points_on_image.png", bbox_inches="tight", dpi=300
        )
        plt.close()
    return DepthAlignmentPipeline.from_config(config).align(
        image,
        predicted_depth,
        sfm_points_camera,
        sfm_points_depth,
        config,
        debug_export_dir,
    )


def get_subsampler(cfg: Config):
    if cfg.mdi.subsample_factor == "adaptive":
        return AdaptiveDepthSubsampler(cfg.mdi.adaptive_subsampling)
    elif isinstance(cfg.mdi.subsample_factor, int):
        return StaticDepthSubsampler(cfg.mdi.subsample_factor)  # noqa: F821
    else:
        raise ValueError(f"Unsupported subsampling factor: {cfg.mdi.subsample_factor}")


def get_pts_from_depth(
    predicted_depth: PredictedDepth,
    image: InputImage,
    parser: Parser | NerfbaselinesParser,
    config: Config,
    device: str,
    debug_export_dir: Optional[Path] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        pts_world: torch.Tensor on device of shape [N, 3] where N is the number of points in the world space
        subsample_mask: torch.Tensor on cpu of shape [N] where N is the number of points in the world space
        P: torch.Tensor on device of shape [3, 4] the projection matrix
    """
    imsize = predicted_depth.depth.T.shape

    R = image.cam2world[:3, :3].T
    C = image.cam2world[:3, 3]
    P = image.K @ R @ torch.hstack([torch.eye(3), -C[:, None]])

    sfm_points = (
        torch.from_numpy(parser.points[parser.point_indices[image.name]])
        .to(device)
        .float()
    )

    cam2world = image.cam2world.to(device).float()
    P = P.to(device).float()
    K = image.K.to(device).float()

    if torch.any(torch.isinf(predicted_depth.depth[predicted_depth.mask])):
        _LOGGER.warning("Encountered infinite depths in predicted depth map.")

    aligned_depth, mask = align_depth(
        image.data,
        config,
        sfm_points,
        P,
        imsize,
        predicted_depth,
        debug_export_dir,
    )

    if torch.any(torch.isinf(aligned_depth[mask])):
        _LOGGER.warning("Encountered negative depths in aligned depth map.")

    # NOTE: get_mask applies the passed in mask as well
    mask = get_subsampler(config).get_mask(image.data, aligned_depth, mask)
    mask &= (aligned_depth >= 0).flatten()

    pts_camera: torch.Tensor = torch.dstack(
        [
            torch.from_numpy(np.mgrid[0 : imsize[0], 0 : imsize[1]].T).to(device),
            aligned_depth,
        ],
    ).reshape(-1, 3)[mask]

    pts_camera[:, 0] = (pts_camera[:, 0] + 0.5) * pts_camera[:, 2]
    pts_camera[:, 1] = (pts_camera[:, 1] + 0.5) * pts_camera[:, 2]

    def transform_camera_to_world_space(camera_homo: torch.Tensor) -> torch.Tensor:
        dense_world = torch.linalg.inv(K) @ camera_homo.reshape((-1, 3)).T
        dense_world = (
            cam2world
            @ torch.vstack(
                [dense_world, torch.ones(dense_world.shape[1], device=cam2world.device)]
            )
        )[:3].T
        return dense_world

    pts_world = transform_camera_to_world_space(pts_camera)

    if debug_export_dir is not None:
        debug_export_point_clouds(
            imsize,
            cam2world,
            P,
            transform_camera_to_world_space,
            sfm_points,
            pts_world,
            parser,
            image.name,
            mask.cpu(),
            image.data,
            debug_export_dir,
        )

    return (pts_world.reshape([-1, 3]).float(), mask.cpu(), P)
