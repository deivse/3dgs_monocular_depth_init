from pathlib import Path
import matplotlib
import numpy as np
import torch
import logging

from gs_init_compare.depth_alignment.config import DepthSegmentationConfig
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


def _create_segmentation(masks, image_shape, degenerate_mask_thresh: float):
    masks_sorted_indices = torch.tensor([mask["area"] for mask in masks]).sort(
        descending=True
    )[1]

    segmentation = np.zeros(image_shape, dtype=int)

    image_area = image_shape[0] * image_shape[1]

    curr_region_id = 1
    for mask_ix in masks_sorted_indices:
        region = masks[mask_ix]["segmentation"]
        region_area = masks[mask_ix]["area"]

        if (float(region_area) / float(image_area)) > degenerate_mask_thresh:
            continue  # Likely degenera

        curr_state_for_region = segmentation[region]
        # calculate percentage of each distinct value in curr_state_for_region
        values, counts = np.unique(curr_state_for_region, return_counts=True)
        largest_overlap_id = values[counts.argmax()].item()

        overlap_fraction = counts.max().astype(float) / region.sum().astype(float)

        if overlap_fraction > 0.75 and largest_overlap_id != UNASSIGNED_OVERLAP_ID:
            segmentation[region] = largest_overlap_id
        else:
            segmentation[region] = curr_region_id
            curr_region_id += 1

    return segmentation


def segment_pred_depth_sam(
    pred_depth: PredictedDepth,
    checkpoint_dir: Path,
    sfm_points_camera_coords: torch.Tensor,
    config: DepthSegmentationConfig,
) -> np.ndarray:
    normals = pred_depth.normal
    depth = pred_depth.depth

    depth_lower_bound = torch.quantile(depth, 0.05)
    depth_upper_bound = torch.quantile(depth, 0.95)
    depth[depth < depth_lower_bound] = depth_lower_bound
    depth[depth > depth_upper_bound] = depth_upper_bound
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    device = normals.device

    mask_generator = SamAutomaticMaskGenerator(_get_sam(checkpoint_dir, device))
    depth_rgb = (
        255.0 * matplotlib.cm.get_cmap("viridis")(depth_norm.cpu().numpy())[:, :, :3]
    )
    masks_depth = mask_generator.generate(depth_rgb.astype(np.uint8))

    if normals is not None:
        normals_rgb = torch.round(127.5 * (normals + 1.0)).cpu().numpy().astype(np.uint8)
        masks_normals = mask_generator.generate(normals_rgb)
        all_masks = masks_normals + masks_depth
    else:
        all_masks = masks_depth

    image_shape = (depth.shape[0], depth.shape[1])
    segmentation = _create_segmentation(
        all_masks, image_shape, config.sam.degenerate_mask_thresh
    )
    # segmentation_depth_only = create_segmentation(masks_depth, image_shape)
    # segmentation_normals_only = create_segmentation(masks_normals, image_shape)
    segmentation = ski.segmentation.expand_labels(segmentation, config.sam.expansion_radius)

    # separate unassigned regions into connected components
    unassigned_mask = segmentation == UNASSIGNED_OVERLAP_ID
    labeled_unassigned, num_features = ski.measure.label(
        unassigned_mask, return_num=True
    )
    base_new_label = segmentation.max() + 1
    for feature in range(1, num_features + 1):
        segmentation[labeled_unassigned == feature] = base_new_label
        base_new_label += 1

    image_area = segmentation.shape[0] * segmentation.shape[1]
    tiny_region_thresh = image_area * config.sam.tiny_region_area_fraction
    # separate small disconnected components within each region into new regions
    # so they can be removed during merging afterwards
    new_segmentation = np.zeros_like(segmentation)
    for label in np.unique(segmentation):
        if label == 0:
            continue
        mask = segmentation == label
        labeled_mask, num_features = ski.measure.label(mask, return_num=True)
        component_areas = np.array(
            [(labeled_mask == feature).sum() for feature in range(1, num_features + 1)]
        )
        base_new_label = new_segmentation.max() + 1
        curr_additional_label = base_new_label + 1
        for feature in range(1, num_features + 1):
            if component_areas[feature - 1] >= tiny_region_thresh:
                new_segmentation[labeled_mask == feature] = base_new_label
            else:
                new_segmentation[labeled_mask == feature] = curr_additional_label
                curr_additional_label += 1

    segmentation = new_segmentation

    return segmentation
