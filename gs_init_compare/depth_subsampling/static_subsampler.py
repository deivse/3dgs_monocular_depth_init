from dataclasses import dataclass

import torch
from .interface import DepthSubsampler


@dataclass
class StaticDepthSubsampler(DepthSubsampler):
    subsample_factor: int

    def get_mask(self, rgb, depth, mask_from_predictor):
        pixel_coords = torch.cartesian_prod(
            torch.arange(depth.shape[0]), torch.arange(depth.shape[1])
        )

        return torch.logical_and(
            torch.logical_and(
                (pixel_coords[:, 0] % self.subsample_factor) == 0,
                (pixel_coords[:, 1] % self.subsample_factor) == 0,
            ).to(mask_from_predictor.device),
            mask_from_predictor.view(-1),
        )
