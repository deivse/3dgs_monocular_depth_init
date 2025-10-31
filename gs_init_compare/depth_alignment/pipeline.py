from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from gs_init_compare.config import Config
from gs_init_compare.depth_alignment.interface import (
    DepthAlignmentResult,
    DepthAlignmentStrategy,
    DepthSegmentationFn,
)
from gs_init_compare.depth_alignment.segmentation.region_margin import (
    calculate_region_margin_mask,
)
from gs_init_compare.depth_alignment.segmentation.region_merging import (
    merge_segmentation_regions,
)
from gs_init_compare.depth_prediction.predictors.depth_predictor_interface import (
    PredictedDepth,
)

LOGGER = logging.getLogger(__name__)


def debug_export_segmentation(
    segmentation: torch.Tensor | np.ndarray,
    segment_deadzone_mask: torch.Tensor | None,
    image: torch.Tensor,
    debug_export_dir: Path,
    suffix: str,
):
    if isinstance(segmentation, torch.Tensor):
        seg_np = segmentation.cpu().numpy()
    else:
        seg_np = segmentation
    # overlay image with slic depth regions for visualization
    depth_region_cmap = ListedColormap(
        plt.cm.get_cmap("tab20").colors[: np.unique(seg_np).shape[0]]
    )

    if np.max(seg_np) != 0:
        region_colors = depth_region_cmap(seg_np / np.max(seg_np))[:, :, :3]
    else:
        region_colors = depth_region_cmap(seg_np)[:, :, :3]
    depth_region_overlay = 0.5 * image.cpu().numpy() + 0.5 * region_colors
    depth_region_overlay = np.clip(depth_region_overlay, 0, 1)

    if segment_deadzone_mask is None:
        segment_deadzone_mask = calculate_region_margin_mask(
            torch.from_numpy(segmentation), 2
        )
    depth_region_overlay[~segment_deadzone_mask.cpu().numpy()] *= 0.5

    debug_export_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(
        debug_export_dir / f"depth_regions_overlay_{suffix}.png", depth_region_overlay
    )


def debug_export_alignment(
    predicted_depth: PredictedDepth,
    result: DepthAlignmentResult,
    debug_export_dir: Path,
):
    debug_export_dir.mkdir(parents=True, exist_ok=True)

    # Get depth values for visualization
    initial_depth = predicted_depth.depth.cpu().numpy()
    aligned_depth = result.aligned_depth.cpu().numpy()
    initial_mask = predicted_depth.mask.cpu().numpy()
    result_mask = result.mask.cpu().numpy()

    # Determine common depth range for consistent colormap
    valid_initial = initial_depth[initial_mask]
    valid_aligned = aligned_depth[result_mask]
    if len(valid_initial) > 0 and len(valid_aligned) > 0:
        depth_min = min(np.min(valid_initial), np.min(valid_aligned))
        depth_max = max(np.max(valid_initial), np.max(valid_aligned))
    else:
        depth_min, depth_max = 0, 1

    # Create visualization for initial depth
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(initial_depth, cmap="viridis", vmin=depth_min, vmax=depth_max)
    # Dim masked pixels by overlaying semi-transparent black
    masked_overlay = np.zeros((*initial_depth.shape, 4))
    masked_overlay[~initial_mask] = [0, 0, 0, 0.5]
    ax.imshow(masked_overlay)
    plt.colorbar(im, ax=ax)
    ax.set_title("Initial Predicted Depth")
    plt.savefig(debug_export_dir / "initial_depth.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Create visualization for aligned depth
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(aligned_depth, cmap="viridis", vmin=depth_min, vmax=depth_max)
    # Dim masked pixels
    masked_overlay = np.zeros((*aligned_depth.shape, 4))
    masked_overlay[~result_mask] = [0, 0, 0, 0.5]
    ax.imshow(masked_overlay)
    plt.colorbar(im, ax=ax)
    ax.set_title("Aligned Depth")
    plt.savefig(debug_export_dir / "aligned_depth.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Create difference heatmap
    depth_diff = aligned_depth - initial_depth
    # Only show differences where both masks are valid
    valid_diff_mask = initial_mask & result_mask
    depth_diff[~valid_diff_mask] = 0

    # Determine symmetric range for diverging colormap
    valid_diff = depth_diff[valid_diff_mask]
    if len(valid_diff) > 0:
        max_abs_diff = np.max(np.abs(valid_diff))
        vmin, vmax = -max_abs_diff, max_abs_diff
    else:
        vmin, vmax = -1, 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(depth_diff, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    # Dim invalid pixels
    masked_overlay = np.zeros((*depth_diff.shape, 4))
    masked_overlay[~valid_diff_mask] = [0, 0, 0, 0.7]
    ax.imshow(masked_overlay)
    plt.colorbar(im, ax=ax)
    ax.set_title("Depth Difference (Aligned - Initial)")
    plt.savefig(debug_export_dir / "depth_difference.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Create multiplicative factor heatmap
    mult_factor = np.ones_like(aligned_depth)
    # Only compute factors where both depths are valid and non-zero
    valid_mult_mask = valid_diff_mask & (initial_depth != 0)
    mult_factor[valid_mult_mask] = (
        aligned_depth[valid_mult_mask] / initial_depth[valid_mult_mask]
    )
    mult_factor[~valid_mult_mask] = 1.0  # Set invalid pixels to neutral factor

    # Determine symmetric range for diverging colormap centered at 1
    valid_mult = mult_factor[valid_mult_mask]
    if len(valid_mult) > 0:
        max_mult = np.max(valid_mult)
        min_mult = np.min(valid_mult)
        # Symmetric range around 1
        max_deviation = max(abs(max_mult - 1), abs(min_mult - 1))
        vmin, vmax = 1 - max_deviation, 1 + max_deviation
    else:
        vmin, vmax = 0.5, 1.5

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mult_factor, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    # Dim invalid pixels
    masked_overlay = np.zeros((*mult_factor.shape, 4))
    masked_overlay[~valid_mult_mask] = [0, 0, 0, 0.7]
    ax.imshow(masked_overlay)
    plt.colorbar(im, ax=ax, label="Multiplicative Factor")
    ax.set_title("Depth Multiplication Factor (Aligned / Initial)")
    plt.savefig(
        debug_export_dir / "depth_mult_factor.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


@dataclass
class DepthAlignmentPipeline:
    config: Config
    segmentation: Optional[DepthSegmentationFn]
    alignment: DepthAlignmentStrategy

    @staticmethod
    def from_config(config: Config):
        segmentation = None
        if config.mdi.alignment.segmenter is not None:
            segmentation = config.mdi.alignment.segmenter.get_implementation()
        alignment = config.mdi.alignment.aligner.get_implementation()
        return DepthAlignmentPipeline(config, segmentation, alignment)

    def align(
        self,
        image: torch.Tensor,
        predicted_depth: PredictedDepth,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        config: Config,
        debug_export_dir: Path | None,
    ):
        device = predicted_depth.depth.device
        num_sfm_points = sfm_points_depth.shape[0]

        if self.segmentation:
            segmentation_config = config.mdi.alignment.segmentation
            # pred_depth: PredictedDepth,
            # checkpoint_dir: Path,
            # sfm_points_camera_coords: torch.Tensor,
            # config: InterpConfig,
            segmentation = self.segmentation(
                predicted_depth,
                Path(config.mdi.cache_dir) / "checkpoints",
                sfm_points_camera_coords,
                segmentation_config,
            )

            if debug_export_dir is not None:
                debug_export_segmentation(
                    segmentation,
                    None,
                    image,
                    debug_export_dir,
                    "before_merge",
                )

            segmentation = merge_segmentation_regions(
                predicted_depth,
                sfm_points_camera_coords,
                segmentation,
                segmentation_config,
            )
            segment_deadzone_mask = calculate_region_margin_mask(
                segmentation, segmentation_config.region_margin
            )

            if debug_export_dir is not None:
                debug_export_segmentation(
                    segmentation,
                    segment_deadzone_mask,
                    image,
                    debug_export_dir,
                    "after_merge",
                )

            region_ids = torch.unique(segmentation[predicted_depth.mask])
            region_sfm_point_indices = []

            if config.mdi.alignment.segmentation.propagate_mask:
                predicted_depth.mask = predicted_depth.mask & segment_deadzone_mask

            sfm_pts_regions = segmentation[
                sfm_points_camera_coords[1], sfm_points_camera_coords[0]
            ]
            sfm_points_segment_mask = segment_deadzone_mask[
                sfm_points_camera_coords[1], sfm_points_camera_coords[0]
            ]

            for region in region_ids:
                region_points = (sfm_pts_regions == region) & sfm_points_segment_mask
                region_sfm_point_indices.append(torch.where(region_points)[0])
        else:
            segmentation = torch.zeros_like(predicted_depth.depth, dtype=torch.int)
            region_ids = torch.tensor([0], device=device)
            region_sfm_point_indices = [torch.arange(num_sfm_points, device=device)]

        INVALID_DEPTH_VAL = -42.0
        out_depth = torch.full_like(predicted_depth.depth, INVALID_DEPTH_VAL)
        out_mask = torch.ones_like(out_depth, dtype=bool)

        for region in region_ids:
            region_sfm_coords = sfm_points_camera_coords[
                :, region_sfm_point_indices[region.item()]
            ]
            region_gt_depth = sfm_points_depth[region_sfm_point_indices[region.item()]]
            region_num_pts = region_sfm_coords.shape[1]

            if region_num_pts == 0:
                # TODO: how come this happens with SAM segmentation and region merging, where region merging should take into account the margin around boundaries?
                LOGGER.error(
                    "No SfM points found in region %s; removing region from output.",
                    region.item(),
                )
                continue

            region_mask = segmentation == region

            region_alignment_result = self.alignment.align(
                predicted_depth,
                region_sfm_coords,
                region_gt_depth,
                config,
                debug_export_dir,
            )

            out_depth[region_mask] = region_alignment_result.aligned_depth[region_mask]
            out_mask[region_mask] &= region_alignment_result.mask[region_mask]

        result = DepthAlignmentResult(
            aligned_depth=out_depth,
            mask=(out_depth != INVALID_DEPTH_VAL) & predicted_depth.mask,
        )

        if debug_export_dir is not None:
            debug_export_alignment(predicted_depth, result, debug_export_dir)

        return result
