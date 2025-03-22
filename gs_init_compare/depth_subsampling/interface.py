import abc

import torch


class DepthSubsampler(abc.ABC):
    def get_mask(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `rgb`   input RGB image `[H, W, 3]`
            `depth` input depth map `[H, W]`
        Returns:
            Boolean sampling mask of same shape as flattened depth - [H * W].
            Indicates which points should be used.
        """
