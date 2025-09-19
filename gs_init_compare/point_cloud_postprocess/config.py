from dataclasses import dataclass

from enum import Enum

from pointcloud_subsampling.subsampling_params import PointCloudSubsamplingParams


class OutlierRemovalMethod(str, Enum):
    off = "none"
    lof = "lof"


@dataclass
class PointCloudPostprocessConfig:
    outlier_removal: OutlierRemovalMethod = OutlierRemovalMethod.off
    lof_num_neighbors: int = 40
    subsample: bool = False
    subsample_params: PointCloudSubsamplingParams = PointCloudSubsamplingParams()
