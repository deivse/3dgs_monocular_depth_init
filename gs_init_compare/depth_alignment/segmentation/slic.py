from pathlib import Path
import numpy as np
import skimage
import torch

from gs_init_compare.depth_alignment.config import DepthSegmentationConfig
from gs_init_compare.depth_prediction.predictors.depth_predictor_interface import (
    PredictedDepth,
)


def segment_pred_depth_slic(
    pred_depth: PredictedDepth,
    checkpoint_dir: Path,
    config: DepthSegmentationConfig,
) -> np.ndarray:
    depth = pred_depth.depth
    mask = pred_depth.mask

    valid_depth = depth[mask]
    pred_depth_norm = (depth - valid_depth.min()) / (
        valid_depth.max() - valid_depth.min() + 1e-8
    )
    pred_depth_norm = pred_depth_norm.cpu().numpy()

    compactness = config.slic.compactness
    num_regions = config.slic.num_regions
    slic_depth_regions = skimage.segmentation.slic(
        pred_depth_norm,
        n_segments=num_regions,
        start_label=0,
        compactness=compactness,
        channel_axis=None,
        mask=mask.cpu().numpy().astype(bool),
    )

    return slic_depth_regions
