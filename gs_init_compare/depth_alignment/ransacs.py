from typing import Callable

from gs_init_compare.config import Config
from gs_init_compare.depth_alignment.config import RansacConfig
from .lstsqrs import align_depth_least_squares
import math
import torch

from .interface import DepthAlignmentStrategy


class DepthAlignmentRansac(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        image: torch.Tensor,
        predicted_depth: torch.Tensor,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        config: Config,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return _align_depth_ransac_generic(
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
            _ransac_loss,
            config.mdi.ransac,
        )


class DepthAlignmentMsac(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        image: torch.Tensor,
        predicted_depth: torch.Tensor,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        config: Config,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return _align_depth_ransac_generic(
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
            _msac_loss,
            config.mdi.ransac,
        )


def _ransac_loss(dists: torch.Tensor, inlier_threshold: float):
    return torch.sum(dists >= inlier_threshold)


def _msac_loss(dists: torch.Tensor, inlier_threshold: float):
    return torch.sum(torch.minimum(dists, torch.full_like(dists, inlier_threshold)))


RansacLossFunc = Callable[[torch.Tensor, float], float]
"""
A function that computes the loss of the alignment between the predicted and ground truth depth maps.

Args:
    squared_distances: The squared distances between the predicted and ground truth depth maps.
    inlier_threshold: The threshold for considering a point an inlier.
Returns: loss
"""


def _required_samples(
    inlier_count: int,
    total_correspondence_count: int,
    min_sample_size: int,
    confidence: float,
):
    # k = log(η)/ log(1 − P_I).
    # P_I ≈ ε^m
    inlier_ratio = inlier_count / total_correspondence_count
    try:
        return math.log(1 - confidence) / math.log(1 - inlier_ratio**min_sample_size)
    except (ZeroDivisionError, ValueError):
        return 0


def _l2_dists_squared(
    h: tuple[float, float], depth: torch.Tensor, gt_depth: torch.Tensor
) -> torch.Tensor:
    return (h[0] * depth + h[1] - gt_depth) ** 2


def _align_depth_ransac_generic(
    depth: torch.Tensor,
    gt_points_camera_coords: torch.Tensor,
    gt_depth: torch.Tensor,
    loss_func: RansacLossFunc,
    config: RansacConfig,
) -> torch.Tensor:
    full_predicted_depth = depth
    depth = depth[gt_points_camera_coords[1], gt_points_camera_coords[0]].flatten()

    SAMPLE_SIZE = 2
    num_samples = depth.shape[0]
    device = depth.device
    depth = torch.vstack(
        [
            depth.reshape(-1),
            torch.ones(num_samples, device=device),
        ]
    )

    h_best: tuple[float, float] | None = None
    loss_best = float("inf")
    inlier_indices_best = torch.empty_like(gt_depth, dtype=bool)
    num_inliers_best = 0

    p = config

    for iteration in range(p.max_iters):
        sample_indices = torch.randint(0, num_samples, (SAMPLE_SIZE,))
        h = align_depth_least_squares(
            depth[:, sample_indices], gt_depth[sample_indices]
        )

        dists = _l2_dists_squared(h, depth[0], gt_depth)
        inlier_indices = dists < p.inlier_threshold
        loss = loss_func(dists, p.inlier_threshold)
        if loss < loss_best:
            h_best = align_depth_least_squares(
                depth[:, inlier_indices], gt_depth[inlier_indices]
            )
            dists = _l2_dists_squared(h_best, depth[0], gt_depth)
            loss_best = loss_func(dists, p.inlier_threshold)
            inlier_indices_best = dists < p.inlier_threshold
            num_inliers_best = torch.sum(inlier_indices_best)

        if (
            _required_samples(num_inliers_best, num_samples, SAMPLE_SIZE, p.confidence)
            <= iteration
            and h_best is not None
            and iteration >= config.min_iters
        ):
            break

    h_best = align_depth_least_squares(
        depth[:, inlier_indices_best], gt_depth[inlier_indices_best]
    )
    inlier_indices_best = (
        torch.abs(h_best[0] * depth[0] + h_best[1] - gt_depth) < p.inlier_threshold
    )
    num_inliers_best = torch.sum(inlier_indices_best)

    #########################################
    h_test = align_depth_least_squares(depth, gt_depth)
    inlier_indices_test = (
        torch.abs(h_test[0] * depth[0] + h_test[1] - gt_depth) < p.inlier_threshold
    )
    num_inliers_test = torch.sum(inlier_indices_test)
    avg_err_best = torch.mean(torch.abs(h_best[0] * depth[0] + h_best[1] - gt_depth))
    avg_err_test = torch.mean(torch.abs(h_test[0] * depth[0] + h_test[1] - gt_depth))
    print(
        f"#(iter): {iteration}, (RAN/M)SAC inliers: {num_inliers_best}, Naive inliers: {num_inliers_test}, (RAN/M)SAC Error: {avg_err_best}, Naive Error: {avg_err_test}"
    )
    ###########################################
    # if avg_err_test < avg_err_best:
    #     print("Naive alignment is better than RANSAC")
    #     print(
    #         f"ransac inlier ratio: {num_inliers_best / num_samples}, naive inlier ratio: {num_inliers_test / num_samples}")
    # else:
    #     print("RANSAC alignment is better than naive")
    #     print(
    #         f"ransac inlier ratio: {num_inliers_best / num_samples}, naive inlier ratio: {num_inliers_test / num_samples}")
    return full_predicted_depth * h_best[0] + h_best[1]
