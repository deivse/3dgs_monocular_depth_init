from pathlib import Path
import matplotlib
import numpy as np
import torch
import logging

from gs_init_compare.depth_alignment.config import InterpConfig
from gs_init_compare.depth_alignment.segmentation.region_margin import calculate_region_margin_mask
from gs_init_compare.depth_prediction.predictors.depth_predictor_interface import (
    PredictedDepth,
)
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from gs_init_compare.utils.download_with_tqdm import download_with_pbar
import skimage as ski

LOGGER = logging.getLogger(__name__)

UNASSIGNED_OVERLAP_ID = 0

_sam = None


def _get_sam(checkpoint_dir: Path, device: torch.device):
    global _sam
    if _sam is None:
        LOGGER.info("Loading SAM model...")
        checkpoint_path = checkpoint_dir / "sam_vit_h_4b8939.pth"
        download_with_pbar(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            checkpoint_path,
        )
        _sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    return _sam.to(device)


def _create_segmentation(masks, image_shape):
    masks_sorted_indices = torch.tensor(
        [mask["area"] for mask in masks]).sort(descending=True)[1]

    segmentation = np.zeros(image_shape, dtype=int)

    for i, mask_ix in enumerate(masks_sorted_indices):
        region = masks[mask_ix]["segmentation"]
        curr_state_for_region = segmentation[region]

        # calculate percentage of each distinct value in curr_state_for_region
        values, counts = np.unique(
            curr_state_for_region, return_counts=True)
        largest_overlap_id = values[counts.argmax()].item()

        overlap_fraction = counts.max().astype(float) / region.sum().astype(float)

        if overlap_fraction > 0.75 and largest_overlap_id != UNASSIGNED_OVERLAP_ID:
            segmentation[region] = largest_overlap_id
        else:
            segmentation[region] = i + 1

    segmentation = ski.segmentation.relabel_sequential(segmentation)[0]
    return segmentation


def segment_pred_depth_sam(
    pred_depth: PredictedDepth,
    checkpoint_dir: Path,
    sfm_points_camera_coords: torch.Tensor,
    config: InterpConfig,
) -> torch.Tensor:
    normals = pred_depth.normal
    depth = pred_depth.depth

    # TODO: handle pred_depth.mask!!!

    depth_lower_bound = torch.quantile(depth, 0.05)
    depth_upper_bound = torch.quantile(depth, 0.95)
    depth[depth < depth_lower_bound] = depth_lower_bound
    depth[depth > depth_upper_bound] = depth_upper_bound
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    device = normals.device

    mask_generator = SamAutomaticMaskGenerator(
        _get_sam(checkpoint_dir, device))
    normals_rgb = torch.round(
        127.5 * (normals + 1.0)).cpu().numpy().astype(np.uint8)
    masks_normals = mask_generator.generate(normals_rgb)
    depth_rgb = 255.0 * \
        matplotlib.cm.get_cmap('viridis')(depth_norm.cpu().numpy())[:, :, :3]
    masks_depth = mask_generator.generate(depth_rgb.astype(np.uint8))

    all_masks = masks_normals + masks_depth

    EXPANSION_RADIUS = 4
    TINY_REGION_IMAGE_AREA_FRACTION = 1e-4
    MIN_NUM_SFM_POINTS_IN_REGION = 8

    image_shape = (depth.shape[0], depth.shape[1])
    segmentation = _create_segmentation(all_masks, image_shape)
    # segmentation_depth_only = create_segmentation(masks_depth, image_shape)
    # segmentation_normals_only = create_segmentation(masks_normals, image_shape)
    segmentation = ski.segmentation.expand_labels(
        segmentation, EXPANSION_RADIUS)

    # separate unassigned regions into connected components
    unassigned_mask = segmentation == UNASSIGNED_OVERLAP_ID
    labeled_unassigned, num_features = ski.measure.label(
        unassigned_mask, return_num=True)
    base_new_label = segmentation.max() + 1
    for feature in range(1, num_features + 1):
        segmentation[labeled_unassigned == feature] = base_new_label
        base_new_label += 1

    image_area = segmentation.shape[0] * segmentation.shape[1]
    tiny_region_thresh = image_area * TINY_REGION_IMAGE_AREA_FRACTION
    # separate small disconnected components within each region into new regions
    # so they can be merged afterwards
    new_segmentation = np.zeros_like(segmentation)
    for label in np.unique(segmentation):
        if label == 0:
            continue
        mask = segmentation == label
        labeled_mask, num_features = ski.measure.label(mask, return_num=True)
        component_areas = np.array([(labeled_mask == feature).sum()
                                    for feature in range(1, num_features + 1)])
        base_new_label = new_segmentation.max() + 1
        curr_additional_label = base_new_label + 1
        for feature in range(1, num_features + 1):
            if component_areas[feature - 1] >= tiny_region_thresh:
                new_segmentation[labeled_mask == feature] = base_new_label
            else:
                new_segmentation[labeled_mask ==
                                 feature] = curr_additional_label
                curr_additional_label += 1

    segmentation = new_segmentation

    rag = ski.graph.rag_boundary(
        segmentation, np.ones_like(segmentation).astype(float))

    margin_mask = calculate_region_margin_mask(
        torch.from_numpy(segmentation).to(device), config, depth.shape)
    sfm_pts_margin_mask = margin_mask[
        sfm_points_camera_coords[1], sfm_points_camera_coords[0]].cpu().numpy()

    region_num_sfm_points = {}
    sfm_points_np = sfm_points_camera_coords.cpu().numpy()
    for i in np.unique(segmentation):
        region_points = (segmentation[
            sfm_points_np[1], sfm_points_np[0]
        ] == i)
        region_points = region_points & sfm_pts_margin_mask
        region_num_sfm_points[i] = region_points.sum().item()

    rename_dict = {}
    while True:
        if min(region_num_sfm_points.values()) >= MIN_NUM_SFM_POINTS_IN_REGION:
            break
        worst_region = min(region_num_sfm_points.keys(),
                           key=lambda key: region_num_sfm_points[key])

        neighbors = np.array(list(rag.neighbors(worst_region)))

        for i in range(neighbors.size):
            while neighbors[i] in rename_dict:
                neighbors[i] = rename_dict[neighbors[i]]

        neighbors = np.unique(neighbors[neighbors != worst_region])
        if neighbors.size == 0:
            # Nothing we can do, remove from consideration
            region_num_sfm_points[worst_region] = float('inf')
            continue

        neighbors_depth_diffs = np.array([
            (pred_depth.depth[segmentation == worst_region].mean() -
             pred_depth.depth[segmentation == n].mean()).abs().item()
            for n in neighbors
        ])
        best_neighbor_region = neighbors[neighbors_depth_diffs.argmin()]
        segmentation[segmentation == worst_region] = best_neighbor_region
        region_num_sfm_points[best_neighbor_region] += region_num_sfm_points[worst_region]

        region_num_sfm_points.pop(worst_region)
        rename_dict[worst_region] = best_neighbor_region

    # relabel sequentially
    segmentation -= segmentation.min()
    segmentation = ski.segmentation.relabel_sequential(segmentation)[0]
    segmentation = torch.from_numpy(segmentation).to(device)

    return segmentation, calculate_region_margin_mask(segmentation, config, depth.shape)
