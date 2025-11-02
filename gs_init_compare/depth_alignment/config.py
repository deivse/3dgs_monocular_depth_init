from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional


class DepthAlignmentStrategyEnum(str, Enum):
    lstsqrs = "lstsqrs"
    ransac = "ransac"
    msac = "msac"
    interp = "interp"

    def get_implementation(self):
        if self == self.lstsqrs:
            from .alignment.lstsqrs import DepthAlignmentLstSqrs

            return DepthAlignmentLstSqrs
        elif self == self.ransac:
            from .alignment.ransacs import DepthAlignmentRansac

            return DepthAlignmentRansac
        elif self == self.msac:
            from .alignment.ransacs import DepthAlignmentMsac

            return DepthAlignmentMsac
        elif self == self.interp:
            from .alignment.interp import DepthAlignmentInterpolate

            return DepthAlignmentInterpolate
        elif self == self.sam:
            from .segmentation.sam import DepthAlignmentSAM

            return DepthAlignmentSAM
        else:
            raise NotImplementedError(f"Unknown depth alignment strategy: {self}")


class DepthSegmentationStrategyEnum(str, Enum):
    slic = "slic"
    sam = "sam"

    def get_implementation(self):
        if self == self.slic:
            from .segmentation.slic import segment_pred_depth_slic

            return segment_pred_depth_slic
        elif self == self.sam:
            from .segmentation.sam import segment_pred_depth_sam

            return segment_pred_depth_sam
        else:
            raise NotImplementedError(f"Unknown depth segmentation strategy: {self}")


@dataclass
class SAMSegmentationconfig:
    use_normals: bool = True
    """
    If true, merge masks for depth and normals (if available), otherwise use only depth masks.
    """

    degenerate_mask_thresh: float = 0.9
    """
    Masks the area of which is above this fraction of the image area are ignored as degenerate.
    """

    expansion_radius: int = 4
    """
    Since SAM masks don't necessarily cover the entire object, expand each mask by this
    radius (in pixels) so merging can properly detect neighboring regions.
    """
    tiny_region_area_fraction: float = 1e-4
    """
    Disconnected sub-regions of a given SAM mask that are smaller than this fraction of the image area
    are assigned new region IDs so they can be removed during merging.
    """


@dataclass
class SLICSegmentationConfig:
    compactness = 0.01
    num_regions = 40


@dataclass
class DepthSegmentationConfig:
    region_margin: int = 10
    """
    Pixels closer than this distance from the depth segmentation boundary are ignored when interpolating scale factors. 
    Helps avoid issues at object boundaries. The value is normalized for image size, for an image of size (H, W), the actual margin is:
        margin = int(segmentation_region_margin * min(H, W) / 480)
    """
    propagate_mask: bool = False
    """
    If segmentation is used, mask out deadzones around segmentation boundaries in the alignment pipeline output mask.
    """

    min_border_grad_threshold: float = 0.0005
    min_sfm_pts_in_region: int = 5

    sam: SAMSegmentationconfig = SAMSegmentationconfig()
    slic: SLICSegmentationConfig = SLICSegmentationConfig()


@dataclass
class RansacConfig:
    inlier_threshold: float = 0.01
    max_iters: int = 2500
    confidence: float = 0.999
    sample_size: int = 4
    min_iters: int = 0


@dataclass
class InterpConfig:
    method: Literal["rbf", "linear"] = "linear"
    """Interpolation method."""
    init: Literal["lstsqrs", "ransac"] | None = "ransac"
    """If set, use this method to get an initial estimate of scale and shift before scale factor interpolation."""

    scale_outlier_removal: bool = True
    """If true, use Local Outlier Factor to remove outliers in scale factors before RBF interpolation"""

    # RBF-specific parameters
    smoothing: float = 0.001
    """RBF smoothing parameter, see torch_rbf doc"""
    kernel: str = "thin_plate_spline"
    """RBF kernel type, see torch_rbf doc"""
    max_rbf_points: int = 5000
    """Maximum number of points to use for RBF interpolation, -1 means use all points"""


@dataclass
class DepthAlignmentConfig:
    # Strategy to align predicted depth to depth of known SfM points.
    segmenter: Optional[DepthSegmentationStrategyEnum] = None
    aligner: DepthAlignmentStrategyEnum = DepthAlignmentStrategyEnum.ransac

    segmentation: DepthSegmentationConfig = DepthSegmentationConfig()

    # Applies to both RANSAC and MSAC.
    ransac: RansacConfig = RansacConfig()
    interp: InterpConfig = InterpConfig()
