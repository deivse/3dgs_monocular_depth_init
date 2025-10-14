from dataclasses import dataclass
from enum import Enum
from typing import Literal


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
    sample_size: int = 2
    min_iters: int = 0


@dataclass
class InterpConfig:
    method: Literal["rbf", "linear"] = "linear"
    """Interpolation method."""
    init: Literal["lstsqrs", "ransac"] | None = "ransac"
    """If set, use this method to get an initial estimate of scale and shift before scale factor interpolation."""

    segmentation: Literal["slic", "sam"] | None = "slic"
    """If not None, use depth segmentation to split the image into regions without large depth discontinuities and align each region separately"""
    segmentation_region_margin: int = 10
    """
    Pixels closer than this distance from the depth segmentation boundary are ignored when interpolating scale factors. 
    Helps avoid issues at object boundaries. The value is normalized for image size, for an image of size (H, W), the actual margin is:
        margin = int(segmentation_region_margin * min(H, W) / 480)
    """
    segmentation_deadzone_mask: bool = False
    """
    If segmentation is used, mask out deadzones around segmentation boundaries in the output mask.
    """

    scale_outlier_removal: bool = True
    """If true, use Local Outlier Factor to remove outliers in scale factors before RBF interpolation"""

    # RBF-specific parameters
    smoothing: float = 0.001
    """RBF smoothing parameter, see torch_rbf doc"""
    kernel: str = "thin_plate_spline"
    """RBF kernel type, see torch_rbf doc"""
    max_rbf_points: int = 5000
    """Maximum number of points to use for RBF interpolation, -1 means use all points"""
