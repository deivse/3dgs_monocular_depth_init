from dataclasses import dataclass


@dataclass
class PointCloudSubsamplingParams:
    max_bbox_aspect_ratio: float = 1.1
    """
    Max ratio of bounding box dimensions (longest/shortest) to allow merging points.
    Higher values allow merging points in more elongated clusters, which decreases
    precision but increases the number of merged points.
    """
    min_extent_multiplier: float = 1.0
    """
    Multiplier for the average minimal Gaussian extent to determine if points can be merged.
    Higher values allow merging points that are further apart, this can be thought of
    as the subsampling "aggressiveness".
    """
