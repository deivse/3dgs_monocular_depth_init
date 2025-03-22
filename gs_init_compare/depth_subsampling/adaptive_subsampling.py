from typing import Tuple, Union
import torch
from gs_init_compare.depth_prediction.utils.image_filtering import (
    spatial_gradient_first_order,
)


def _map_to_range(tensor: torch.Tensor, range_start=0.0, range_end=1.0):
    tensor = tensor - tensor.min()
    tensor /= tensor.max()
    return (range_end - range_start) * tensor + range_start


def calculate_downsample_factor_map(
    rgb: torch.Tensor,
    tile_size: int = 20,
    possible_subsample_factors: Tuple[int] = (5, 10),
    grad_approx_gauss_sigma: float = 1.2,
) -> torch.Tensor:
    """
    Args:
        `rgb`                        input RGB image `[H, W, 3]`
        `possible_subsample_factors` range of possible subsample factors in increasing order
        `tile_size`                  size of tiles on which subsampling factor is uniform
        `grad_approx_gauss_sigma`    sigma for gaussian kernel used to approximate
                                     gradient of the image
    """
    h, w, _ = rgb.shape
    color_grad = (
        spatial_gradient_first_order(
            rgb.permute(2, 0, 1)[None], sigma=grad_approx_gauss_sigma
        )
        .sum(1)
        .sum(1)
    )
    df = _map_to_range(color_grad.abs())
    df = torch.nn.functional.adaptive_avg_pool2d(
        df,
        [h // tile_size, w // tile_size],
    )

    df = _map_to_range(
        1 - df,
        possible_subsample_factors[0],
        possible_subsample_factors[-1],
    ).squeeze()

    # Clamp to closest value in range
    range = torch.tensor(possible_subsample_factors)
    dists = torch.abs(df[:, :, None] - range[None, None, :])
    return range[torch.argmin(torch.abs(dists), dim=-1)]


def get_sample_mask(
    downsample_factor_map: torch.Tensor,
    image_size: Union[torch.Size, Tuple[int, int]],
) -> torch.Tensor:
    """
    Generates a tensor of boolean values indicating which pixel indices should be sampled
    based on the provided downsample factor map and the desired image size.
    Args:
        downsample_factor_map (torch.Tensor): A tensor representing the downsample factors
            for each pixel in the original image.
        image_size (Union[torch.Size, Tuple[int, int]]): The size of the image to which the
            downsample factor map should be interpolated.
    Returns:
        torch.Tensor: A boolean 1D tensor of length width * hight which can be used to index, e.g. img.view(-1, 3)
    """
    per_pixel_df: torch.Tensor = (
        torch.nn.functional.interpolate(
            downsample_factor_map[None, None].to(float), size=image_size, mode="nearest"
        )
        .squeeze()
        .to(int)
    )
    pixel_coords = torch.cartesian_prod(
        torch.arange(per_pixel_df.shape[0]), torch.arange(per_pixel_df.shape[1])
    )

    per_pixel_df[per_pixel_df == 0] = 1
    return torch.logical_and(
        (pixel_coords[:, 0] % per_pixel_df.view(-1)) == 0,
        (pixel_coords[:, 1] % per_pixel_df.view(-1)) == 0,
    )
