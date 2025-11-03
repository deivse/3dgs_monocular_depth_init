from dataclasses import dataclass
import logging
import time
from typing import Dict
import numpy as np
import torch
import skimage as ski
import scipy.ndimage

from gs_init_compare.depth_alignment.config import DepthSegmentationConfig
from gs_init_compare.depth_alignment.segmentation.region_margin import (
    calculate_region_margin_mask,
    get_actual_margin_size,
)
from gs_init_compare.depth_prediction.predictors.depth_predictor_interface import (
    PredictedDepth,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class _RegionInfo:
    num_sfm_pts: int
    mean_border_grad: float


def merge_segmentation_regions(
    pred_depth: PredictedDepth,
    sfm_points_camera_coords: torch.Tensor,
    segmentation: np.ndarray,
    config: DepthSegmentationConfig,
):
    if np.unique(segmentation).size == 1:
        # only one region, return zeros (in case that region didn't have ID 0)
        segmentation = np.zeros_like(segmentation)
        return segmentation, np.ones_like(segmentation, dtype=bool)

    start = time.perf_counter()

    # Ensures that RAG includes link to region that had id 0 which ski treats as background.
    segmentation = segmentation + 1
    rag = ski.graph.rag_boundary(segmentation, np.ones_like(segmentation).astype(float))

    depth = pred_depth.depth
    depth_norm = depth / (depth.max() - depth.min() + 1e-8)
    depth_gradient_squared = (
        torch.gradient(depth_norm)[0] ** 2 + torch.gradient(depth_norm)[1] ** 2
    )

    sfm_points_np = sfm_points_camera_coords.cpu().numpy()

    adjusted_region_margin = get_actual_margin_size(depth.shape, config.region_margin)
    pred_depth_mask_np = pred_depth.mask.cpu().numpy()

    def num_sfm_pts_in_region(region_id):
        # NOTE: this erosion-based implementation doesn't give the exact same result
        # as calculate_region_margin_mask but it's close enough and faster to compute.
        region_mask = scipy.ndimage.binary_erosion(
            segmentation == region_id, iterations=adjusted_region_margin
        )
        return (region_mask & pred_depth_mask_np)[
            sfm_points_np[1], sfm_points_np[0]
        ].sum()

    def region_border_thick(region_id):
        region_mask = segmentation == region_id
        dilated = ski.morphology.binary_dilation(region_mask)
        eroded = ski.morphology.binary_erosion(region_mask)
        return dilated != eroded

    def border_grad_mean(region_id):
        return depth_gradient_squared[region_border_thick(region_id)].mean().item()

    def border_grad_mean_boundary(region_id_a, region_id_b):
        border_mask = region_border_thick(region_id_a) & region_border_thick(
            region_id_b
        )
        return depth_gradient_squared[border_mask].mean().item()

    region_data: Dict[int, _RegionInfo] = {
        i: _RegionInfo(num_sfm_pts_in_region(i), border_grad_mean(i))
        for i in np.unique(segmentation)
    }
    rename_dict = {}

    while len(region_data) > 1:
        min_grad_region = min(
            region_data.keys(), key=lambda i: region_data[i].mean_border_grad
        )
        min_sfm_pts_region = min(
            region_data.keys(), key=lambda i: region_data[i].num_sfm_pts
        )

        min_grad_pass = (
            region_data[min_grad_region].mean_border_grad
            >= config.min_border_grad_threshold
        )
        min_sfm_pass = (
            region_data[min_sfm_pts_region].num_sfm_pts >= config.min_sfm_pts_in_region
        )
        if min_grad_pass and min_sfm_pass:
            break

        target_region = min_grad_region if not min_grad_pass else min_sfm_pts_region

        neighbors = np.array(list(rag.neighbors(target_region)))

        for i in range(neighbors.size):
            og_val = neighbors[i]
            while neighbors[i] in rename_dict:
                neighbors[i] = rename_dict[neighbors[i]]
            if og_val in rename_dict:
                # Create "shortcut"
                rename_dict[og_val] = neighbors[i]

        neighbors = np.unique(neighbors[neighbors != target_region])
        if neighbors.size == 0:
            # Nothing we can do, remove from consideration
            LOGGER.error(
                "Segmentation region %d is chosen for merging but is disconnected."
            )
            region_data[target_region].mean_border_grad = float("inf")
            region_data[target_region].num_sfm_pts = float("inf")
            continue

        border_depth_grad_factors = np.array(
            [
                border_grad_mean_boundary(target_region, neighbor)
                for neighbor in neighbors
            ],
            dtype=float,
        )
        best_neighbor = neighbors[border_depth_grad_factors.argmin()]

        segmentation[segmentation == target_region] = best_neighbor
        region_data[best_neighbor].mean_border_grad = border_grad_mean(best_neighbor)
        region_data[best_neighbor].num_sfm_pts = num_sfm_pts_in_region(best_neighbor)

        # TODO: better merging
        for neighbor in neighbors:
            if neighbor != best_neighbor:
                rag.add_edge(best_neighbor, neighbor)

        region_data.pop(target_region)
        rename_dict[target_region] = best_neighbor

    # relabel sequentially
    segmentation -= segmentation.min()
    segmentation = ski.segmentation.relabel_sequential(segmentation)[0]
    segmentation = torch.from_numpy(segmentation).to(depth.device)

    LOGGER.info(f"Merging regions took {time.perf_counter() - start:.2f} seconds")
    return segmentation


# @dataclass
# class _RegionInfoOld:
#     num_sfm_pts: int
#     area: int


# def merge_segmentation_regions_old(
#     pred_depth: PredictedDepth,
#     sfm_points_camera_coords: torch.Tensor,
#     segmentation: np.ndarray,
#     config: DepthSegmentationConfig,
# ):
#     if np.unique(segmentation).size == 1:
#         # only one region, return zeros (in case that region didn't have ID 0)
#         segmentation = np.zeros_like(segmentation)
#         return segmentation, np.ones_like(segmentation, dtype=bool)

#     rag = ski.graph.rag_boundary(segmentation, np.ones_like(segmentation).astype(float))

#     depth = pred_depth.depth
#     margin_mask = calculate_region_margin_mask(
#         torch.from_numpy(segmentation).to(depth.device), config.region_margin
#     )
#     sfm_pts_mask = (
#         (
#             margin_mask[sfm_points_camera_coords[1], sfm_points_camera_coords[0]]
#             & pred_depth.mask[sfm_points_camera_coords[1], sfm_points_camera_coords[0]]
#         )
#         .cpu()
#         .numpy()
#     )

#     region_data: Dict[int, _RegionInfoOld] = {}
#     sfm_points_np = sfm_points_camera_coords.cpu().numpy()
#     for i in np.unique(segmentation):
#         region_points = segmentation[sfm_points_np[1], sfm_points_np[0]] == i
#         region_points = region_points & sfm_pts_mask
#         region_data[i] = _RegionInfoOld(
#             num_sfm_pts=int(region_points.sum().item()),
#             area=int((segmentation == i).sum()),
#         )

#     image_area = int(depth.shape[0] * depth.shape[1])
#     rename_dict = {}

#     depth_norm = depth / (depth.max() - depth.min() + 1e-8)
#     depth_gradient_squared = (
#         torch.gradient(depth_norm)[0] ** 2 + torch.gradient(depth_norm)[1] ** 2
#     )
#     while True:
#         region_with_least_sfm_pts = min(
#             region_data.keys(), key=lambda i: region_data[i].num_sfm_pts
#         )
#         region_with_smallest_area = min(
#             region_data.keys(), key=lambda i: region_data[i].area
#         )

#         min_sfm_pts_satisfied = (
#             region_data[region_with_least_sfm_pts].num_sfm_pts
#             >= config.min_sfm_pts_in_region
#         )
#         min_area_satisfied = (
#             float(region_data[region_with_smallest_area].area) / float(image_area)
#             >= config.min_region_area_fraction
#         )

#         if min_sfm_pts_satisfied and min_area_satisfied:
#             break

#         if not min_sfm_pts_satisfied:
#             worst_region = region_with_least_sfm_pts
#         else:
#             worst_region = region_with_smallest_area

#         neighbors = np.array(list(rag.neighbors(worst_region)))

#         for i in range(neighbors.size):
#             og_val = neighbors[i]
#             while neighbors[i] in rename_dict:
#                 neighbors[i] = rename_dict[neighbors[i]]
#             if og_val in rename_dict:
#                 # Create "shortcut"
#                 rename_dict[og_val] = neighbors[i]

#         neighbors = np.unique(neighbors[neighbors != worst_region])
#         if neighbors.size == 0:
#             # Nothing we can do, remove from consideration
#             region_data[worst_region].area = float("inf")
#             region_data[worst_region].num_sfm_pts = float("inf")
#             continue

#         border_depth_grad_factors = np.empty(neighbors.size, dtype=float)
#         for i, neighbor in enumerate(neighbors):
#             # Find border pixels between worst_region and neighbor
#             worst_mask = segmentation == worst_region
#             neighbor_mask = segmentation == neighbor

#             # double-dilate both masks to find adjacent pixels with some extra margin
#             worst_dilated = ski.morphology.binary_dilation(
#                 ski.morphology.binary_dilation(worst_mask)
#             )
#             neighbor_dilated = ski.morphology.binary_dilation(
#                 ski.morphology.binary_dilation(neighbor_mask)
#             )

#             # Border pixels are where dilated regions overlap
#             border_mask = worst_dilated & neighbor_dilated

#             border_depth_grad_factors[i] = depth_gradient_squared[
#                 border_mask
#             ].sum().item() / min(float(border_mask.sum()), 1.0)

#         best_neighbor_region = neighbors[border_depth_grad_factors.argmin()]
#         segmentation[segmentation == worst_region] = best_neighbor_region
#         region_data[best_neighbor_region].num_sfm_pts += region_data[
#             worst_region
#         ].num_sfm_pts
#         region_data[best_neighbor_region].area += region_data[worst_region].area
#         for neighbor in neighbors:
#             if neighbor != best_neighbor_region:
#                 rag.add_edge(best_neighbor_region, neighbor)
#         region_data.pop(worst_region)
#         rename_dict[worst_region] = best_neighbor_region

#     # relabel sequentially
#     segmentation -= segmentation.min()
#     segmentation = ski.segmentation.relabel_sequential(segmentation)[0]
#     segmentation = torch.from_numpy(segmentation).to(depth.device)
#     return segmentation
