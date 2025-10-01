from dataclasses import dataclass
from enum import Enum


class DepthAlignmentStrategyEnum(str, Enum):
    lstsqrs = "lstsqrs"
    ransac = "ransac"
    msac = "msac"
    interp = "interp"

    def get_implementation(self):
        if self == self.lstsqrs:
            from .lstsqrs import DepthAlignmentLstSqrs

            return DepthAlignmentLstSqrs
        elif self == self.ransac:
            from .ransacs import DepthAlignmentRansac

            return DepthAlignmentRansac
        elif self == self.msac:
            from .ransacs import DepthAlignmentMsac

            return DepthAlignmentMsac
        elif self == self.interp:
            from .interp import DepthAlignmentInterpolate

            return DepthAlignmentInterpolate
        else:
            raise NotImplementedError(f"Unknown depth alignment strategy: {self}")


@dataclass
class RansacConfig:
    inlier_threshold: float = 0.1
    max_iters: int = 2500
    confidence: float = 0.99
    min_iters: int = 0


@dataclass
class InterpConfig:
    # TODO: after initial testing, change defaults to:
    # lstsqrs_init: False
    # smoothing: 0.0001
    lstsqrs_init: bool = True
    smoothing: float = 0.001
    kernel: str = "thin_plate_spline"  # see torch_rbf doc for options
    segmentation: bool = False
