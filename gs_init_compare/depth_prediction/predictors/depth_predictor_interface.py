from abc import ABCMeta, abstractmethod
from typing import Optional, NamedTuple

import torch


class PredictedDepth(NamedTuple):
    depth: torch.Tensor
    """ Float tensor of shape (H, W) """
    mask: torch.Tensor
    """ Bool tensor indicating valid pixels. (H, W) """
    depth_confidence: Optional[torch.Tensor] = None
    """ Optional float tensor indicating confidence of each pixel. (H, W) """
    normal: Optional[torch.Tensor] = None
    """ Optional float tensor of shape (H, W, 3) """
    normal_confidence: Optional[torch.Tensor] = None
    """ Optional float tensor indicating confidence of each normal vector. (H, W, 3) """


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
    torch.serialization.add_safe_globals([PredictedDepth])


class DepthPredictor(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config, device):
        pass

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
