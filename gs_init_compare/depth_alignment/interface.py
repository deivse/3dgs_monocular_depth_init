import abc
from pathlib import Path

import torch


class DepthAlignmentStrategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def align(
        cls,
        predicted_depth: torch.Tensor,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        config,  # : Config,
        debug_export_dir: Path | None,
    ) -> torch.Tensor:
        """
        Estimate the alignment between predicted and ground truth depth maps and return the aligned depth map.

        Args:
            predicted_depth: The predicted depth map. Shape: [Width, Height]
            sfm_points_camera_coords: The (y, x) (in that order!) coordinates of the SfM points in the camera frame. Shape: [2, NumPoints]
            sfm_points_depth: The depth of the SfM points. Shape: [NumPoints]
            ransac_config: Configuration for RANSAC (will be ignored if not applicable).
        """
