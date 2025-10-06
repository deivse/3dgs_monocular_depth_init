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
    # smoothing: 0.0001 for tanksandtemples, 0.001 for mipnerf360 - middle ground 0.0005?
    # kernel: "thin_plate_spline" (NOT TESTED YET)
    # segmentation: True
    # segmentation_region_margin: ? (10 for bonsai)
    # segmentation_deadzone_mask: False

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
    segmentation_deadzone_mask: bool = False
    """
    If segmentation is used, mask out deadzones around segmentation boundaries in the output mask.
    This can help avoid invalid reprojections around object boundaries, since the predicted depth resolution is lower than the input image resolution.
    """
    max_rbf_points: int = 5000
    """Maximum number of points to use for RBF interpolation, -1 means use all points"""
    scale_outlier_removal: bool = False
    """If true, use Local Outlier Factor to remove outliers in scale factors before RBF interpolation"""
