import skimage
import torch

from gs_init_compare.depth_alignment.config import InterpConfig
from gs_init_compare.depth_alignment.segmentation.region_margin import calculate_region_margin_mask


def segment_pred_depth_slic(
    predicted_depth: torch.Tensor,
    mask: torch.Tensor,
    config: InterpConfig,
) -> torch.Tensor:
    valid_pred_depth = predicted_depth[mask]
    pred_depth_norm = (predicted_depth - valid_pred_depth.min()) / (
        valid_pred_depth.max() - valid_pred_depth.min() + 1e-8
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
        mask=mask.cpu().numpy().astype(bool),
    )

    slic_depth_regions = torch.from_numpy(slic_depth_regions).to(predicted_depth.device)
    mask = calculate_region_margin_mask(slic_depth_regions, config, predicted_depth.shape)

    return slic_depth_regions, mask
