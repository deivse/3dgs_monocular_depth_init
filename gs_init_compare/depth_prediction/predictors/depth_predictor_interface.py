from abc import ABCMeta, abstractmethod
from typing import Optional, NamedTuple

import torch


class PredictedDepth(NamedTuple):
    depth: torch.Tensor
    """ Float tensor of shape (H, W) """
    mask: torch.Tensor
    """ Bool tensor indicating valid pixels. (H, W) """


class PredictedPoints(NamedTuple):
    points: torch.Tensor
    """ Float tensor of shape (H, W, 3) """
    mask: Optional[torch.Tensor]
    """
    Bool tensor indicating valid pixels. (H, W)
    """


class CameraIntrinsics(NamedTuple):
    K: torch.Tensor

    @property
    def fx(self):
        return self.K[0, 0].item()

    @property
    def fy(self):
        return self.K[1, 1].item()

    @property
    def cx(self):
        return self.K[0, 2].item()

    @property
    def cy(self):
        return self.K[1, 2].item()


if torch.__version__ >= "2.4.0":
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

    def predict_depth(
        self, img: torch.Tensor, intrinsics: CameraIntrinsics
    ) -> PredictedDepth:
        """
        Predict depth from a single image.

        Args:
            img: tensor of shape (H, W, 3).
            intrinsics: Camera intrinsics from sparse reconstruction.

        Returns:
            Depth map.
        """
        raise NotImplementedError

    def predict_points(self, img, intrinsics: CameraIntrinsics) -> PredictedPoints:
        """
        Predict 3D point cloud from a single image.

        Args:
            img: Image.
            intrinsics: Camera intrinsics from sparse reconstruction.

        Returns:
            points: 3D points in the camera coordinate system.
            valid_point_indices: Indices of valid points.
        """
        raise NotImplementedError
