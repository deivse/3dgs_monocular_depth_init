from pathlib import Path
from matplotlib.colors import ListedColormap
import numpy as np
import skimage
import torch
import logging

from gs_init_compare.config import Config
from gs_init_compare.depth_alignment.config import InterpConfig
from gs_init_compare.depth_alignment.lstsqrs import DepthAlignmentLstSqrs
from .interface import DepthAlignmentStrategy
from torchrbf import RBFInterpolator
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def segment_depth_regions(
    predicted_depth: torch.Tensor, image: torch.Tensor, debug_export_dir: Path | None
):
    pred_depth_norm = (predicted_depth - predicted_depth.min()) / (
        predicted_depth.max() - predicted_depth.min() + 1e-8
    )
    pred_depth_norm = pred_depth_norm.cpu().numpy()

    compactness = 0.0001
    num_regions = 5
    slic_depth_regions = skimage.segmentation.slic(
        pred_depth_norm,
        n_segments=num_regions,
        start_label=0,
        compactness=compactness,
        channel_axis=None,
    )

    if debug_export_dir is not None:
        # overlay image with slic depth regions for visualization
        depth_region_cmap = ListedColormap(
            plt.cm.get_cmap("tab20").colors[: np.unique(slic_depth_regions).shape[0]]
        )

        if np.max(slic_depth_regions) != 0:
            region_colors = depth_region_cmap(
                slic_depth_regions / np.max(slic_depth_regions)
            )[:, :, :3]
        else:
            region_colors = depth_region_cmap(slic_depth_regions)[:, :, :3]
        depth_region_overlay = 0.5 * image.cpu().numpy() + 0.5 * region_colors
        depth_region_overlay = np.clip(depth_region_overlay, 0, 1)
        debug_export_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(
            debug_export_dir / "slic_depth_regions_overlay.png", depth_region_overlay
        )
    return slic_depth_regions


def align_depth_interpolate(
    image: torch.Tensor,
    depth: torch.Tensor,
    sfm_points_camera_coords: torch.Tensor,
    gt_depth: torch.Tensor,
    debug_export_dir: Path | None,
    config: InterpConfig,
):
    """
    Args:
        depth: torch.Tensor of shape (Width, Height)
        sfm_points_camera_coords: torch.Tensor of shape (2, N)
                where N is the number of points, the first row is y and the second row is x.
        gt_depth: torch.Tensor of shape (N,)
    """
    H, W = depth.shape

    def rbf_interpolation(coords: torch.Tensor, values: torch.Tensor):
        interpolator = RBFInterpolator(
            coords,
            values,
            smoothing=config.smoothing,
            kernel=config.kernel,
            device=depth.device,
        )

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

    if config.lstsqrs_init:
        unaligned = DepthAlignmentLstSqrs.align(
            image,
            predicted_depth=depth,
            sfm_points_camera_coords=sfm_points_camera_coords,
            sfm_points_depth=gt_depth,
        )
    else:
        unaligned = depth

    if config.segmentation:
        region_map = torch.from_numpy(
            segment_depth_regions(depth, image, debug_export_dir)
        ).to(depth.device)
        # Filter SfM points to keep only one point per depth region
        region_ids = torch.unique(region_map)
        region_sfm_point_indices = []

        # TODO: 1. Try adding deadzone around region boundaries (requires alignment step outputting mask)
        #          Can be implemented by blurring the segmentation map, then points near boundaries will not match any integer region id)

        # TODO: 2. Support masks from predictor
        # TODO: 3. SfM point outlier rejection before rbf step

        for region in region_ids:
            region_points = (
                region_map[sfm_points_camera_coords[1], sfm_points_camera_coords[0]]
                == region
            )
            region_sfm_point_indices.append(torch.where(region_points)[0])
    else:
        region_map = torch.zeros_like(depth, dtype=torch.int)
        region_ids = torch.tensor([0], device=depth.device)
        region_sfm_point_indices = [torch.arange(sfm_points_camera_coords.shape[1])]

    INVALID_SCALE_VAL = -42
    final_scale_map = torch.full_like(depth, INVALID_SCALE_VAL)
    for region in region_ids:
        # limit number of points for RBF to avoid OOM
        region_sfm_pts_camera_coords = sfm_points_camera_coords[
            :, region_sfm_point_indices[region.item()]
        ]
        region_sfm_pts_camera_coords_norm = torch.hstack(
            [
                (region_sfm_pts_camera_coords.float()[0] / (W - 1.0))[:, None],
                (region_sfm_pts_camera_coords.float()[1] / (H - 1.0))[:, None],
            ]
        )
        region_gt_depth = gt_depth[region_sfm_point_indices[region.item()]]
        region_num_pts = region_sfm_pts_camera_coords_norm.shape[0]

        MAX_RBF_POINTS = 5000
        if region_num_pts > MAX_RBF_POINTS:
            indices = torch.randperm(
                region_sfm_pts_camera_coords_norm.shape[0],
                device=depth.device,
            )[:MAX_RBF_POINTS]
            region_sfm_pts_camera_coords = region_sfm_pts_camera_coords[:, indices]
            region_gt_depth = region_gt_depth[indices]
            region_sfm_pts_camera_coords_norm = region_sfm_pts_camera_coords_norm[
                indices
            ]

        if region_num_pts == 0:
            LOGGER.error(
                "No SfM points found in region %s; skipping RBF interpolation.",
                region.item(),
            )
            continue

        try:
            region_scale_map = rbf_interpolation(
                region_sfm_pts_camera_coords_norm,
                region_gt_depth
                / unaligned[
                    region_sfm_pts_camera_coords[1], region_sfm_pts_camera_coords[0]
                ],
            )
            final_scale_map[region_map == region] = region_scale_map[
                region_map == region
            ]
        except Exception as e:
            LOGGER.warning(
                "RBF interpolation failed for region %s with error %s; using median scale instead of RBF interpolation.",
                region.item(),
                e,
            )
            final_scale_map[region_map == region] = (
                region_gt_depth
                / unaligned[
                    region_sfm_pts_camera_coords[1], region_sfm_pts_camera_coords[0]
                ]
            ).median()

    aligned_depth = final_scale_map * unaligned
    aligned_depth[
        final_scale_map == INVALID_SCALE_VAL
    ] = -1  # Will be cleaned up later. TODO: output mask instead, this is awful code

    if debug_export_dir is not None:
        print("Saving images to dir", debug_export_dir)

        vmin = min(unaligned.min(), aligned_depth.min()).item()
        vmax = max(unaligned.max(), aligned_depth.max()).item()
        debug_export_dir.mkdir(parents=True, exist_ok=True)

        # Save pre-RBF-alignment depth map
        # Create copies for visualization with red pixels where depth < 1
        lstsqrs_vis = unaligned.cpu().numpy().copy()
        aligned_vis = aligned_depth.cpu().numpy().copy()

        # Set pixels where depth < 1 to red (this will override the colormap)
        # We'll use a custom approach with RGB arrays
        import matplotlib.cm as cm

        # Normalize the depth values for colormap
        norm_lstsqrs = (lstsqrs_vis - vmin) / (vmax - vmin)
        norm_aligned = (aligned_vis - vmin) / (vmax - vmin)

        # Apply colormap
        lstsqrs_rgb = cm.plasma(norm_lstsqrs)
        aligned_rgb = cm.plasma(norm_aligned)

        # Set red color where depth < 0
        red_mask_lstsqrs = lstsqrs_vis < 0
        red_mask_aligned = aligned_vis < 0

        lstsqrs_rgb[red_mask_lstsqrs] = [1, 0, 0, 1]  # Red
        aligned_rgb[red_mask_aligned] = [1, 0, 0, 1]  # Red

        # Save pre-RBF-alignment depth map with colorbar
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(lstsqrs_rgb)
        ax.set_title("Least Squares Aligned Depth")
        # Create a colorbar with the original depth values
        sm = plt.cm.ScalarMappable(
            cmap=cm.plasma, norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Depth")
        # Add note about red pixels
        ax.text(
            0.02,
            0.98,
            "Red: depth < 0",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment="top",
        )
        pre_rbf_path = Path(debug_export_dir) / "lstsqrs_depth.png"
        plt.savefig(pre_rbf_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Save rbf scale with colorbar
        fig, ax = plt.subplots(figsize=(10, 8))
        rbf_scale_np = final_scale_map.cpu().numpy()
        im = ax.imshow(rbf_scale_np, cmap="viridis")
        ax.set_title("RBF Scale Factor")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Scale Factor")

        # Plot SfM points as small crosses colored by their depth
        sfm_x = sfm_points_camera_coords[0].cpu().numpy()
        sfm_y = sfm_points_camera_coords[1].cpu().numpy()
        sfm_depths = gt_depth.cpu().numpy()

        scatter = ax.scatter(
            sfm_x, sfm_y, marker="x", c="black", s=10, alpha=0.8, linewidths=1
        )

        rbf_scale_path = Path(debug_export_dir) / "rbf_scale.png"
        plt.savefig(rbf_scale_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Save final aligned depth map with colorbar
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(aligned_rgb)
        ax.set_title("RBF Aligned Depth")
        # Create a colorbar with the original depth values
        sm = plt.cm.ScalarMappable(
            cmap=cm.plasma, norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Depth")
        # Add note about red pixels
        ax.text(
            0.02,
            0.98,
            "Red: depth < 0",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment="top",
        )
        aligned_depth_path = Path(debug_export_dir) / "rbf_aligned_depth.png"
        plt.savefig(aligned_depth_path, dpi=150, bbox_inches="tight")
        plt.close()

    return aligned_depth


class DepthAlignmentInterpolate(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        image: torch.Tensor,
        predicted_depth: torch.Tensor,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        config: Config,
        debug_export_dir: Path | None,
    ) -> torch.Tensor:
        return align_depth_interpolate(
            image,
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
            debug_export_dir,
            config.mdi.interp,
        )
