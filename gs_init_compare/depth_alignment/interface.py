import abc

import torch

from gs_init_compare.depth_alignment.config import RansacConfig


class DepthAlignmentStrategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def align(
        cls,
        predicted_depth: torch.Tensor,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        ransac_config: RansacConfig,
    ) -> torch.Tensor:
        """
        Estimate the alignment between predicted and ground truth depth maps and return the aligned depth map.

        Args:
            predicted_depth: The predicted depth map. Shape: [Width, Height]
            sfm_points_camera_coords: The (y, x) (in that order!) coordinates of the SfM points in the camera frame. Shape: [2, NumPoints]
            sfm_points_depth: The depth of the SfM points. Shape: [NumPoints]
            ransac_config: Configuration for RANSAC (will be ignored if not applicable).
        """
