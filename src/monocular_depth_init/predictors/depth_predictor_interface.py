from abc import ABCMeta, abstractmethod
from typing import Tuple

from PIL import Image

import torch


class DepthPredictor(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config, device):
        pass

    @abstractmethod
    def can_predict_points_directly(self) -> bool:
        """
        Returns whether the predictor can predict 3D points directly or only depth.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the predictor.
        """

    def predict_depth(self, img: Image.Image, fx: float, fy: float) -> torch.Tensor:
        """
        Predict depth from a single image.

        Args:
            img: Image.
            fx: Focal length in x direction.
            fy: Focal length in y direction.

        Returns:
            Depth map.
        """
        raise NotImplementedError

    def predict_3d_point_cloud(self, img, fx, fy) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict 3D point cloud from a single image.

        Args:
            img: Image.
            fx: Focal length in x direction.
            fy: Focal length in y direction.

        Returns:
            points: 3D points in the camera coordinate system.
            valid_point_indices: Indices of valid points.
        """
        raise NotImplementedError
