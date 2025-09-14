from dataclasses import dataclass

from enum import Enum


class OutlierRemovalMethod(str, Enum):
    off = "none"
    lof = "lof"


@dataclass
class PointCloudPostprocessConfig:
    outlier_removal: OutlierRemovalMethod = OutlierRemovalMethod.off
    nyquist_subsample: bool = True
    nyquist_subsample_factor: float = 1
