import torch
from gs_init_compare.depth_alignment.config import DepthSegmentationConfig
from gs_init_compare.utils.image_filtering import box_blur2d


def snap_to_int_if_close(x: torch.Tensor):
    """
    Round values that are within `tol` of an integer,
    leave everything else unchanged.
    """
    nearest = x.round()
    mask = torch.isclose(x, nearest)
    return torch.where(mask, nearest, x)


def get_actual_margin_size(image_shape, region_margin) -> int:
    KERNEL_REFERENCE_IMSIZE = 1297
    return int(region_margin * max(image_shape) / KERNEL_REFERENCE_IMSIZE)


def calculate_region_margin_mask(
    region_map: torch.Tensor, region_margin: int
) -> torch.Tensor:
    if region_margin == 0:
        return torch.ones_like(region_map, dtype=torch.bool)
    
    kernel_size = 2 * get_actual_margin_size(region_map.shape, region_margin) + 1

    region_map_blurred = box_blur2d(region_map[None, None].float(), ksize=kernel_size)[
        0, 0
    ]
    region_map_blurred = snap_to_int_if_close(region_map_blurred)
    return region_map_blurred == region_map
