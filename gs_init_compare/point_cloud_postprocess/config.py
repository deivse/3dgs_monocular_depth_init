from dataclasses import dataclass

from enum import Enum


class OutlierRemovalMethod(str, Enum):
    off = "none"
    lof = "lof"


@dataclass
class PointCloudPostprocessConfig:
    knn_outlier_removal: bool = True
    outlier_removal: OutlierRemovalMethod = OutlierRemovalMethod.lof
    octree_clustering: bool = True

    def requires_descriptors(self) -> bool:
        return False
