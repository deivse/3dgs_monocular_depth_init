import gc
import numpy as np
import torch

from gs_init_compare.depth_alignment.lstsqrs import DepthAlignmentLstSqrs
from .interface import DepthAlignmentStrategy
from torchrbf import RBFInterpolator


def align_depth_interpolate(
    depth: torch.Tensor, sfm_points_camera_coords: torch.Tensor, gt_depth: torch.Tensor
):
    """
    Args:
        depth: torch.Tensor of shape (Width, Height)
        sfm_points_camera_coords: torch.Tensor of shape (2, N)
                where N is the number of points, the first row is y and the second row is x.
        gt_depth: torch.Tensor of shape (N,)
    """
    H, W = depth.shape

    def rbf_interpolation(coords, values):
        interpolator = RBFInterpolator(
            coords,
            values,
            smoothing=0.001,
            kernel="thin_plate_spline",
            device=depth.device,
        )

        # TODO: instead of having a fixed downsample factor, adapt it based on image size
        desired_width = 256
        factor = max(W / desired_width, 1)
        query_width = int(W / factor)
        query_height = int(H / factor)

        # Query coordinates
        x = torch.linspace(0, 1, query_width, device=depth.device)
        y = torch.linspace(0, 1, query_height, device=depth.device)
        grid_points = torch.meshgrid(x, y, indexing="ij")
        grid_points = torch.stack(grid_points, dim=-1).reshape(-1, 2)

        # Query RBF on grid points
        interpolated = interpolator(grid_points)
        interpolated = interpolated.reshape(query_width, query_height)[None, None, :, :]
        return torch.nn.functional.interpolate(
            interpolated,
            size=(W, H),
            mode="bilinear",
            align_corners=True,
        )[0, 0, :, :].T

    # first compute approximate offset using least squares
    lstsqrs_aligned = DepthAlignmentLstSqrs.align(
        predicted_depth=depth,
        sfm_points_camera_coords=sfm_points_camera_coords,
        sfm_points_depth=gt_depth,
    )

    coords = torch.hstack(
        [
            (sfm_points_camera_coords[0] / (W - 1))[:, None],
            (sfm_points_camera_coords[1] / (H - 1))[:, None],
        ]
    )

    # limit number of points for RBF to avoid OOM
    MAX_RBF_POINTS = 5000
    if coords.shape[0] > MAX_RBF_POINTS:
        indices = torch.randperm(coords.shape[0], device=coords.device)[:MAX_RBF_POINTS]
        sfm_points_camera_coords = sfm_points_camera_coords[:, indices]
        coords = coords[indices]
        gt_depth = gt_depth[indices]

    # Compute interpolations
    rbf_scale = rbf_interpolation(
        coords,
        gt_depth
        / lstsqrs_aligned[sfm_points_camera_coords[1], sfm_points_camera_coords[0]],
    )

    # TODO: somehow get ransac-like outlier rejection while keeping the "adaptive alignment"?
    # TODO: study failure cases - when does this make things worse?

    aligned_depth = rbf_scale * lstsqrs_aligned
    return aligned_depth


class DepthAlignmentInterpolate(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        predicted_depth: torch.Tensor,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return align_depth_interpolate(
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
        )
