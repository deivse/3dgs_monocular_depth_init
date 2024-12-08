from abc import ABCMeta, abstractmethod
from typing import Optional, NamedTuple

from PIL import Image

import torch


class PredictedDepth(NamedTuple):
    depth: torch.Tensor
    """ Float tensor of shape (H, W) """
    mask: Optional[torch.Tensor]
    """ Bool tensor indicating valid pixels. (H, W) """


class PredictedPoints(NamedTuple):
    points: torch.Tensor
    """ Float tensor of shape (H, W, 3) """
    mask: Optional[torch.Tensor]
    """
    Bool tensor indicating valid pixels. (H, W)
    """


torch.serialization.add_safe_globals([PredictedDepth, PredictedPoints])


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

    def predict_depth(self, img: torch.Tensor, fx: float, fy: float) -> PredictedDepth:
        """
        Predict depth from a single image.

        Args:
            img: tensor of shape (H, W, 3).
            fx: Focal length in x direction.
            fy: Focal length in y direction.

        Returns:
            Depth map.
        """
        raise NotImplementedError

    def predict_points(self, img, fx, fy) -> PredictedPoints:
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
