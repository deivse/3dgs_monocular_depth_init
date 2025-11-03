from pathlib import Path
from typing import Callable

from gs_init_compare.config import Config
from gs_init_compare.depth_alignment.config import RansacConfig
from gs_init_compare.depth_prediction.predictors.depth_predictor_interface import (
    PredictedDepth,
)
from .lstsqrs import align_depth_least_squares
import math
import torch

from ..interface import DepthAlignmentResult, DepthAlignmentStrategy


class DepthAlignmentRansac(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        predicted_depth: PredictedDepth,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        config: Config,
        debug_export_dir: Path | None = None,
        *args,
        **kwargs,
    ) -> DepthAlignmentResult:
        return _align_depth_ransac_generic(
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
            _ransac_loss,
            config.mdi.alignment.ransac,
            debug_export_dir,
        )


class DepthAlignmentMsac(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        predicted_depth: PredictedDepth,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        config: Config,
        debug_export_dir: Path | None = None,
        *args,
        **kwargs,
    ) -> DepthAlignmentResult:
        return _align_depth_ransac_generic(
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
            _msac_loss,
            config.mdi.alignment.ransac,
            debug_export_dir,
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
    predicted_depth: PredictedDepth,
    gt_points_camera_coords: torch.Tensor,
    gt_depth: torch.Tensor,
    loss_func: RansacLossFunc,
    config: RansacConfig,
    debug_export_dir: Path | None = None,
) -> DepthAlignmentResult:
    full_predicted_depth = predicted_depth.depth
    depth = predicted_depth.depth[
        gt_points_camera_coords[1], gt_points_camera_coords[0]
    ].flatten()

    num_samples = depth.shape[0]
    device = depth.device
    depth = torch.vstack(
        [
            depth.reshape(-1),
            torch.ones(num_samples, device=device),
        ]
    )

    h_best_lo: tuple[float, float] | None = None
    inlier_indices_best_lo = torch.empty_like(gt_depth, dtype=bool)
    num_inliers_best_lo = 0
    loss_best_lo = float("inf")
    loss_best_sample = float("inf")

    p = config

    for iteration in range(p.max_iters):
        sample_indices = torch.randperm(num_samples)[: config.sample_size]
        h_sample = align_depth_least_squares(
            depth[:, sample_indices], gt_depth[sample_indices]
        )

        dists_sample = _l2_dists_squared(h_sample, depth[0], gt_depth)
        inlier_indices_sample = dists_sample < p.inlier_threshold

        loss_sample = loss_func(dists_sample, p.inlier_threshold)

        if loss_sample < loss_best_sample:
            h_lo = align_depth_least_squares(
                depth[:, inlier_indices_sample], gt_depth[inlier_indices_sample]
            )
            dists_lo = _l2_dists_squared(h_lo, depth[0], gt_depth)
            loss_lo = loss_func(dists_lo, p.inlier_threshold)
            if loss_lo < loss_best_lo:
                h_best_lo = h_lo
                loss_best_lo = loss_lo
                loss_best_sample = loss_sample
                inlier_indices_best_lo = dists_lo < p.inlier_threshold
                num_inliers_best_lo = torch.sum(inlier_indices_best_lo)

        if (
            _required_samples(
                num_inliers_best_lo, num_samples, config.sample_size, p.confidence
            )
            <= iteration
            and h_best_lo is not None
            and iteration >= config.min_iters
        ):
            break

    if debug_export_dir is not None:
        h_test = align_depth_least_squares(depth, gt_depth)
        inlier_indices_test = (
            torch.abs(h_test[0] * depth[0] + h_test[1] - gt_depth) < p.inlier_threshold
        )
        num_inliers_test = torch.sum(inlier_indices_test)
        avg_err_best = torch.mean(
            torch.abs(h_best_lo[0] * depth[0] + h_best_lo[1] - gt_depth)
        )
        avg_err_test = torch.mean(
            torch.abs(h_test[0] * depth[0] + h_test[1] - gt_depth)
        )

        debug_export_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_export_dir / "ransac_log.txt", "a") as f:
            f.write(
                f"#(iter): {iteration}, (RAN/M)SAC inliers: {num_inliers_best_lo}, Naive inliers: {num_inliers_test}, (RAN/M)SAC Error: {avg_err_best}, Naive Error: {avg_err_test}\n"
            )
    print(
        f"[RANSAC] Iterations: {iteration}, Inliers: {num_inliers_best_lo}/{num_samples}, best scale: {h_best_lo[0]}, best shift: {h_best_lo[1]}",
    )

    return DepthAlignmentResult(
        aligned_depth=full_predicted_depth * h_best_lo[0] + h_best_lo[1],
        mask=predicted_depth.mask,
    )
