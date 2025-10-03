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
    # kernel: "thin_plate_spline" (NOT TESTED YET)
    # segmentation: True
    # segmentation_region_margin: ? (10 for bonsai)

    lstsqrs_init: bool = True
    """If true, first use least squares to pre-align depth including offset, and then use RBF interpolation to refine the alignment."""
    smoothing: float = 0.001
    """RBF smoothing parameter, see torch_rbf doc"""
    kernel: str = "thin_plate_spline"
    """See torch_rbf doc for options"""
    segmentation: bool = False
    """If true, use segmentation to split the image into regions and align each region separately"""
    segmentation_region_margin: int = 0
    """Half kernel size for box blur, 0 means no blurring"""
