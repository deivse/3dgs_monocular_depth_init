from enum import IntEnum
from pathlib import Path
from typing import NamedTuple
from matplotlib.colors import ListedColormap
import numpy as np
import skimage
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
import torch
import logging

from gs_init_compare.config import Config
from gs_init_compare.depth_alignment.config import InterpConfig
from gs_init_compare.depth_alignment.lstsqrs import DepthAlignmentLstSqrs
from gs_init_compare.utils.image_filtering import box_blur2d, gaussian_filter2d
from .interface import DepthAlignmentResult, DepthAlignmentStrategy
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


def rbf_interpolation(
    coords: torch.Tensor,
    values: torch.Tensor,
    config: InterpConfig,
    device: torch.device,
    W: int,
    H: int,
) -> torch.Tensor:
    interpolator = RBFInterpolator(
        coords,
        values,
        smoothing=config.smoothing,
        kernel=config.kernel,
        device=device,
    )

    desired_width = 256
    factor = max(W / desired_width, 1)
    query_width = int(W / factor)
    query_height = int(H / factor)

    # Query coordinates
    x = torch.linspace(0, 1, query_width, device=device)
    y = torch.linspace(0, 1, query_height, device=device)
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


def pick_rbf_point_subset(
    num_points,
    max_points,
    sfm_pts_camera_coords,
    gt_depth,
    sfm_pts_camera_coords_norm,
    device,
):
    indices = torch.randperm(
        num_points,
        device=device,
    )[:max_points]
    return (
        sfm_pts_camera_coords[:, indices],
        gt_depth[indices],
        sfm_pts_camera_coords_norm[indices],
    )


def snap_to_int(x: torch.Tensor):
    """
    Round values that are within `tol` of an integer,
    leave everything else unchanged.
    """
    nearest = x.round()
    mask = torch.isclose(x, nearest)
    return torch.where(mask, nearest, x)


class OutlierType(IntEnum):
    REGULAR = 0
    SCALE_ONLY = 1
    POSITION_ONLY = 2
    BOTH = 3


class OutlierClassification(NamedTuple):
    scale_only_outliers: torch.Tensor
    both_outliers: torch.Tensor
    position_only_outliers: torch.Tensor
    regular: torch.Tensor


def scale_factor_outlier_removal(
    coords: torch.Tensor, scales: torch.Tensor, debug_export_dir: Path | None
):
    K_lof = 10
    clf = LocalOutlierFactor(n_neighbors=K_lof, n_jobs=-1)
    coords_np = coords.cpu().numpy()
    pred_pts_only = clf.fit_predict(coords_np)
    position_outliers_np = pred_pts_only == -1

    K_scale_knn = 5
    model = NearestNeighbors(n_neighbors=K_scale_knn + 1, metric="euclidean").fit(
        coords_np
    )
    knn_distances, knn_indices = model.kneighbors(coords_np)
    knn_distances = knn_distances[:, 1:]  # remove self-distance
    knn_indices = knn_indices[:, 1:]  # remove self-index

    knn_median_scale = torch.median(scales[knn_indices], dim=1).values
    scale_diff = torch.abs(scales - knn_median_scale)
    scale_diff_threshold = torch.quantile(scale_diff, 0.99)  # can be tuned

    scale_outliers = scale_diff > scale_diff_threshold
    position_outliers = torch.from_numpy(position_outliers_np).to(scale_outliers)

    return OutlierClassification(
        scale_only_outliers=scale_outliers & ~position_outliers,
        both_outliers=scale_outliers & position_outliers,
        position_only_outliers=position_outliers & ~scale_outliers,
        regular=~(scale_outliers | position_outliers),
    )


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
    num_sfm_pts = sfm_points_camera_coords.shape[1]
    device = depth.device

    unaligned = depth
    out_mask = torch.ones_like(depth, dtype=torch.bool)

    if config.lstsqrs_init:
        unaligned, out_mask = DepthAlignmentLstSqrs.align(
            image,
            predicted_depth=depth,
            sfm_points_camera_coords=sfm_points_camera_coords,
            sfm_points_depth=gt_depth,
        )

    # TODO: 1. Try adding deadzone around region boundaries (requires alignment step outputting mask)
    #          Can be implemented by blurring the segmentation map, then points near boundaries will not match any integer region id)
    #          * What if we output the deadzones as a mask and also prevent unprojection of points from those regions?
    #            Even if alignment is perfect, they will have some blur in the depth which will prevent accurate unprojection.
    #       2. SfM point outlier rejection before rbf step? - Can still help because outliers can be inside depth regions as well (e.g. objects with holes)
    #       3. Figure out in what cases RBF fails and design a better fallback (currently just median scale)
    #       4. Support masks from predictor

    if config.segmentation:
        region_map = torch.from_numpy(
            segment_depth_regions(depth, image, debug_export_dir)
        ).to(device)
        # Filter SfM points to keep only one point per depth region
        region_ids = torch.unique(region_map)
        region_sfm_point_indices = []

        region_map_blurred = region_map
        if config.segmentation_region_margin > 0:
            # TODO: dynamic kernel size based on image size?
            kernel_size = 2 * config.segmentation_region_margin + 1
            region_map_blurred = box_blur2d(
                region_map[None, None].float(), ksize=kernel_size
            )[0, 0]
            region_map_blurred = snap_to_int(region_map_blurred)
            if config.segmentation_deadzone_mask:
                out_mask[region_map_blurred != region_map] = False

        for region in region_ids:
            region_points = (
                region_map_blurred[
                    sfm_points_camera_coords[1], sfm_points_camera_coords[0]
                ]
                == region
            )
            region_sfm_point_indices.append(torch.where(region_points)[0])
    else:
        region_map = torch.zeros_like(depth, dtype=torch.int)
        region_ids = torch.tensor([0], device=device)
        region_sfm_point_indices = [torch.arange(num_sfm_pts)]

    global_outlier_type = torch.zeros(num_sfm_pts, dtype=torch.int, device=device)
    INVALID_SCALE_VAL = -42
    final_scale_map = torch.full_like(depth, INVALID_SCALE_VAL)
    for region in region_ids:
        region_sfm_coords = sfm_points_camera_coords[
            :, region_sfm_point_indices[region.item()]
        ]
        region_sfm_coords_norm = torch.hstack(
            [
                (region_sfm_coords.float()[0] / (W - 1.0))[:, None],
                (region_sfm_coords.float()[1] / (H - 1.0))[:, None],
            ]
        )
        region_gt_depth = gt_depth[region_sfm_point_indices[region.item()]]
        region_num_pts = region_sfm_coords_norm.shape[0]

        # limit number of points for RBF to avoid OOM
        if config.max_rbf_points != -1 and region_num_pts > config.max_rbf_points:
            (
                region_sfm_coords,
                region_gt_depth,
                region_sfm_coords_norm,
            ) = pick_rbf_point_subset(
                region_num_pts,
                config.max_rbf_points,
                region_sfm_coords,
                region_gt_depth,
                region_sfm_coords_norm,
                device,
            )

        if region_num_pts == 0:
            # TODO: is it better to skip or use global least squares scale for the whole depth map
            LOGGER.error(
                "No SfM points found in region %s; skipping RBF interpolation.",
                region.item(),
            )
            continue

        region_mask = region_map == region

        try:
            scale_factors = (
                region_gt_depth / unaligned[region_sfm_coords[1], region_sfm_coords[0]]
            )
            if config.scale_outlier_removal:
                outlier_type = scale_factor_outlier_removal(
                    region_sfm_coords.T, scale_factors, debug_export_dir
                )
                outlier_mask = outlier_type.scale_only_outliers
                if outlier_mask.sum() > 0:
                    LOGGER.info(
                        "Removed %d/%d points as outliers from region %s using LOF",
                        outlier_mask.sum().item(),
                        region_num_pts,
                        region.item(),
                    )
                # TODO: UNCOMMENTT!!!
                scale_factors = scale_factors[~outlier_mask]
                region_sfm_coords_norm = region_sfm_coords_norm[~outlier_mask]
                global_outlier_type[
                    region_sfm_point_indices[region][outlier_type.scale_only_outliers]
                ] = OutlierType.SCALE_ONLY
                global_outlier_type[
                    region_sfm_point_indices[region][outlier_type.both_outliers]
                ] = OutlierType.BOTH
                global_outlier_type[
                    region_sfm_point_indices[region][
                        outlier_type.position_only_outliers
                    ]
                ] = OutlierType.POSITION_ONLY
                global_outlier_type[
                    region_sfm_point_indices[region][outlier_type.regular]
                ] = OutlierType.REGULAR

            region_scale_map = rbf_interpolation(
                region_sfm_coords_norm,
                scale_factors,
                config,
                device,
                W,
                H,
            )
            final_scale_map[region_mask] = region_scale_map[region_mask]
        except Exception as e:
            LOGGER.warning(
                "RBF interpolation failed for region %s with error %s; using median scale instead of RBF interpolation.",
                region.item(),
                e,
            )
            final_scale_map[region_mask] = (
                region_gt_depth / unaligned[region_sfm_coords[1], region_sfm_coords[0]]
            ).median()

    aligned_depth = final_scale_map * unaligned
    out_mask = out_mask & (final_scale_map != INVALID_SCALE_VAL)

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
        ax.imshow(image.cpu().numpy())
        im = ax.imshow(rbf_scale_np, cmap="viridis", alpha=0.75)
        ax.set_title("RBF Scale Factor")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Scale Factor")

        sfm_x = sfm_points_camera_coords[0].cpu().numpy()
        sfm_y = sfm_points_camera_coords[1].cpu().numpy()

        # Visualize SfM points with different styles depending on outlier type
        try:
            # Helper to scatter a subset
            def _scatter(mask, color, marker, label, z):
                if mask.numel() == 0 or mask.sum() == 0:
                    return
                ax.scatter(
                    sfm_x[mask.cpu().numpy()],
                    sfm_y[mask.cpu().numpy()],
                    marker=marker,
                    c=color,
                    s=18,
                    alpha=0.9,
                    linewidths=0.8,
                    label=label,
                    zorder=z,
                )

            _scatter(
                global_outlier_type == OutlierType.REGULAR, "black", "x", "regular", 5
            )
            _scatter(
                global_outlier_type == OutlierType.SCALE_ONLY,
                "red",
                "o",
                "scale outlier",
                6,
            )
            _scatter(
                global_outlier_type == OutlierType.POSITION_ONLY,
                "orange",
                "s",
                "pos outlier",
                7,
            )
            _scatter(
                global_outlier_type == OutlierType.BOTH,
                "magenta",
                "D",
                "both outlier",
                8,
            )

            ax.legend(loc="upper right", fontsize=8)
        except Exception:
            # Fallback: single-style scatter if anything goes wrong
            ax.scatter(
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

    return DepthAlignmentResult(aligned_depth, out_mask)


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
