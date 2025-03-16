import abc
from dataclasses import dataclass

import torch


@dataclass
class DepthAlignmentParams:
    h: torch.Tensor

    @property
    def scale(self):
        return self.h[0]

    @property
    def shift(self):
        return self.h[1]


class DepthAlignmentStrategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def estimate_alignment(
        cls, predicted_depth: torch.Tensor, gt_depth: torch.Tensor
    ) -> DepthAlignmentParams:
        """
        Estimate the alignment between predicted and ground truth depth maps.

        Args:
            predicted_depth: The predicted depth map. Shape: [NumPoints]
            gt_depth: The ground truth depth map. Shape: [NumPoints]
        """
