from enum import IntEnum
from pathlib import Path
from typing import NamedTuple
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
import torch
import logging

from gs_init_compare.config import Config
from gs_init_compare.depth_alignment.alignment.lstsqrs import DepthAlignmentLstSqrs
from gs_init_compare.depth_alignment.config import InterpConfig
from gs_init_compare.depth_alignment.segmentation import segment_pred_depth_sam, segment_pred_depth_slic
from gs_init_compare.depth_alignment.alignment.ransacs import DepthAlignmentRansac
from gs_init_compare.depth_prediction.predictors.depth_predictor_interface import (
    PredictedDepth,
)
from ..interface import DepthAlignmentResult, DepthAlignmentStrategy
from torchrbf import RBFInterpolator
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def rbf_interpolation(
    coords: torch.Tensor,
    values: torch.Tensor,
    config: InterpConfig,
    device: torch.device,
    W: int,
    H: int,
) -> torch.Tensor:
    coords_norm = torch.hstack(
        [
            (coords.float()[0] / (W - 1.0))[:, None],
            (coords.float()[1] / (H - 1.0))[:, None],
        ]
    )

    interpolator = RBFInterpolator(
        coords_norm,
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
    interpolated = interpolated.reshape(query_width, query_height)[
        None, None, :, :]
    return torch.nn.functional.interpolate(
        interpolated,
        size=(W, H),
        mode="bilinear",
        align_corners=True,
    )[0, 0, :, :].T


def linear_interpolation(
    coords: torch.Tensor,
    values: torch.Tensor,
    config: InterpConfig,
    device: torch.device,
    W: int,
    H: int,
) -> torch.Tensor:
    coords_np = coords.T.cpu().numpy()
    values_np = values.cpu().numpy()

    # add values at the corners to stabilize interpolation
    corner_coords = np.array([[0, 0], [0, H - 1], [W - 1, 0], [W - 1, H - 1]])
    corner_indices = np.arange(coords_np.shape[0], coords_np.shape[0] + 4)
    coords_np = np.vstack((coords_np, corner_coords))
    values_np = np.hstack((values_np, np.empty(4, dtype=values_np.dtype)))

    dt = Delaunay(coords_np)
    for corner_ix in corner_indices:
        indptr, indices = dt.vertex_neighbor_vertices
        neighbors = indices[indptr[corner_ix]: indptr[corner_ix + 1]]
        # exclude other corners
        neighbors = np.setdiff1d(neighbors, corner_indices)
        distances = np.linalg.norm(
            coords_np[neighbors] - coords_np[corner_ix], axis=1)
        weights = 1.0 / (distances + 1e-8)
        weights /= np.sum(weights)
        corner_value = np.sum(values_np[neighbors] * weights)
        if np.isnan(corner_value):
            corner_value = np.median(values_np[neighbors])
        values_np[corner_ix] = corner_value

    X = np.linspace(0, W - 1, W)
    Y = np.linspace(0, H - 1, H)
    X, Y = np.meshgrid(X, Y)
    interp = LinearNDInterpolator(
        dt, values_np, fill_value=np.median(values_np))
    return torch.from_numpy(interp(X, Y)).to(values)


def interpolate_scale(
    coords: torch.Tensor,
    values: torch.Tensor,
    config: InterpConfig,
    device: torch.device,
    W: int,
    H: int,
) -> torch.Tensor:
    if config.method == "rbf":
        return rbf_interpolation(coords, values, config, device, W, H)
    elif config.method == "linear":
        return linear_interpolation(coords, values, config, device, W, H)
    else:
        raise ValueError(f"Unknown interpolation method: {config.method}")


def pick_rbf_point_subset(
    num_points,
    max_points,
    sfm_pts_camera_coords,
    sfm_depth,
    device,
):
    indices = torch.randperm(
        num_points,
        device=device,
    )[:max_points]
    return (
        sfm_pts_camera_coords[:, indices],
        sfm_depth[indices],
        indices,
    )


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
    K_scale_knn = 5

    num_pts = coords.shape[0]
    if num_pts < min(K_lof + 1, K_scale_knn + 1):
        return OutlierClassification(
            scale_only_outliers=torch.zeros(num_pts, dtype=torch.bool),
            both_outliers=torch.zeros(num_pts, dtype=torch.bool),
            position_only_outliers=torch.zeros(num_pts, dtype=torch.bool),
            regular=torch.ones(num_pts, dtype=torch.bool),
        )

    clf = LocalOutlierFactor(n_neighbors=K_lof, n_jobs=-1)
    coords_np = coords.cpu().numpy()
    pred_pts_only = clf.fit_predict(coords_np)
    position_outliers_np = pred_pts_only == -1

    model = NearestNeighbors(n_neighbors=K_scale_knn + 1, metric="euclidean").fit(
        coords_np
    )
    knn_distances, knn_indices = model.kneighbors(coords_np)

    # remove self-distance/index (first column)
    knn_distances = knn_distances[:, 1:]
    knn_indices = knn_indices[:, 1:]
    knn_median_scale = torch.median(scales[knn_indices], dim=1).values
    scale_diff = torch.abs(scales - knn_median_scale)
    scale_diff_threshold = torch.quantile(scale_diff, 0.99)
    scale_outliers = scale_diff > scale_diff_threshold

    position_outliers = torch.from_numpy(
        position_outliers_np).to(scale_outliers)

    return OutlierClassification(
        scale_only_outliers=scale_outliers & ~position_outliers,
        both_outliers=scale_outliers & position_outliers,
        position_only_outliers=position_outliers & ~scale_outliers,
        regular=~(scale_outliers | position_outliers),
    )


def initial_alignment(
    image: torch.Tensor,
    predicted_depth: PredictedDepth,
    sfm_points_camera_coords: torch.Tensor,
    gt_depth: torch.Tensor,
    config: Config,
    debug_export_dir: Path | None = None,
) -> DepthAlignmentResult:
    if config.mdi.interp.init is None:
        return predicted_depth.depth, predicted_depth.mask
    if config.mdi.interp.init == "lstsqrs":
        return DepthAlignmentLstSqrs.align(
            image,
            predicted_depth,
            sfm_points_camera_coords,
            gt_depth,
            config,
            debug_export_dir,
        )
    elif config.mdi.interp.init == "ransac":
        return DepthAlignmentRansac.align(
            image,
            predicted_depth,
            sfm_points_camera_coords,
            gt_depth,
            config,
            debug_export_dir,
        )
    else:
        raise ValueError(
            f"Unknown interp alignment init method: {config.mdi.interp.init}"
        )


def segment_depth_regions(
    pred_depth: PredictedDepth,
    sfm_points_camera_coords: torch.Tensor,
    image: torch.Tensor,
    config: Config,
    debug_export_dir: Path | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    interp_config = config.mdi.interp
    if interp_config.segmentation == "sam":
        segmentation, mask = segment_pred_depth_sam(
            pred_depth,
            checkpoint_dir=Path(config.mdi.cache_dir) / "checkpoints",
            sfm_points_camera_coords=sfm_points_camera_coords,
            config=config.mdi.interp
        )
    elif interp_config.segmentation == "slic":
        segmentation, mask = segment_pred_depth_slic(
            pred_depth.depth,
            pred_depth.mask,
            config.mdi.interp
        )
    else:
        raise ValueError(
            f"Unknown segmentation method: {interp_config.segmentation_method}"
        )

    if debug_export_dir is not None:
        seg_np = segmentation.cpu().numpy()
        # overlay image with slic depth regions for visualization
        depth_region_cmap = ListedColormap(
            plt.cm.get_cmap("tab20").colors[: np.unique(seg_np).shape[0]]
        )

        if np.max(seg_np) != 0:
            region_colors = depth_region_cmap(
                seg_np / np.max(seg_np)
            )[:, :, :3]
        else:
            region_colors = depth_region_cmap(seg_np)[:, :, :3]
        depth_region_overlay = 0.5 * image.cpu().numpy() + 0.5 * region_colors
        depth_region_overlay = np.clip(depth_region_overlay, 0, 1)

        depth_region_overlay[~mask.cpu().numpy()] = 0.0

        debug_export_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(
            debug_export_dir / "depth_regions_overlay.png", depth_region_overlay
        )

    return segmentation, mask


def align_depth_interpolate(
    predicted_depth: PredictedDepth,
    sfm_points_camera_coords: torch.Tensor,
    gt_depth: torch.Tensor,
    config: Config,
    debug_export_dir: Path | None,
):
    """
    Args:
        depth: torch.Tensor of shape (Width, Height)
        sfm_points_camera_coords: torch.Tensor of shape (2, N)
                where N is the number of points, the first row is y and the second row is x.
        gt_depth: torch.Tensor of shape (N,)
    """
    H, W = predicted_depth.depth.shape
    num_sfm_pts = sfm_points_camera_coords.shape[1]
    device = predicted_depth.depth.device
    interp_config = config.mdi.alignment.interp

    # limit number of points for RBF to avoid OOM
    if (
        interp_config.method == "rbf"
        and interp_config.max_rbf_points != -1
        and num_sfm_pts > interp_config.max_rbf_points
    ):
        (
            sfm_points_camera_coords,
            gt_depth,
        ) = pick_rbf_point_subset(
            num_sfm_pts,
            interp_config.max_rbf_points,
            sfm_points_camera_coords,
            gt_depth,
            device,
        )

    scale_factors = (
        gt_depth /
        predicted_depth.depth[sfm_points_camera_coords[1],
                              sfm_points_camera_coords[0]]
    )
    if interp_config.scale_outlier_removal:
        outlier_type = scale_factor_outlier_removal(
            sfm_points_camera_coords.T, scale_factors, debug_export_dir
        )
        outlier_mask = outlier_type.scale_only_outliers
        if outlier_mask.sum() > 0:
            LOGGER.info(
                "Removed %d/%d scale outlier points.",
                outlier_mask.sum().item(),
                num_sfm_pts
            )
        scale_factors = scale_factors[~outlier_mask]
        sfm_points_camera_coords = sfm_points_camera_coords[:, ~outlier_mask]

    try:
        scale_map = interpolate_scale(
            sfm_points_camera_coords,
            scale_factors,
            interp_config,
            device,
            W,
            H,
        )
    except Exception as e:
        LOGGER.warning(
            "Scale factor interpolation failed; using median scale instead of interpolation.",
            e,
        )
        scale_map = scale_factors.median()

    return DepthAlignmentResult(scale_map * predicted_depth.depth, predicted_depth.mask)


class DepthAlignmentInterpolate(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        predicted_depth: PredictedDepth,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        config: Config,
        debug_export_dir: Path | None,
    ) -> DepthAlignmentResult:
        return align_depth_interpolate(
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
            config,
            debug_export_dir,
        )
