from dataclasses import dataclass


@dataclass
class PointCloudPostprocessConfig:
    voxel_subsample: bool = False
    voxel_size_wrt_scene_extent: float = 2e-3
    outlier_removal: bool = False
