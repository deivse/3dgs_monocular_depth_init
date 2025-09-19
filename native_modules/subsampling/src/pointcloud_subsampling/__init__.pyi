"""
3DGS Monocular Depth Initialization Pointcloud Subsampling Module
-----------------------

.. currentmodule:: pointcloud_subsampling

.. autosummary::
    :toctree: _generate

    subsample_pointcloud
"""

import numpy
from .subsampling_params import PointCloudSubsamplingParams

def subsample_pointcloud(
    points: numpy.ndarray,
    rgbs: numpy.ndarray,
    intrinsic_matrices: list[numpy.ndarray],
    camera_2_world_matrices: list[numpy.ndarray],
    image_sizes: numpy.ndarray,
    params: PointCloudSubsamplingParams,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Merges points that are closer together than their average minimal Gaussian extent,
    which is defined as the min Gaussian size so that it can contribute to a signal below the Nyquist frequency in at least one image).
    This is done using an algorithm similar to top-down KD-tree construction with spatial median splits and round robin axis selection,
    except no actual data structure is built, and recursion stops once points are close enough to be merged.

    1. A tight AABB is computed for the points in the current region -> AABB_tight.
    2. aspect_ratio := the smallest aspect ratio between the original_AABB (built iteratively via median splits) and the tight_AABB.
    3. avg_min_ext := the average minimal Gaussian extent of the points in the current region.
    4. IF max_dimension(tight_AABB) <= avg_min_ext * `min_extent_multiplier` AND aspect_ratio <= `max_bbox_aspect_ratio`, THEN
       merge all points in the current region into a single point at their mean position with their mean
       color.
    5. ELSE IF number of points in the current region <= 1, THEN return the points as-is.
    6. ELSE
       split the points in the current region along the longest axis of the original_AABB at their spatial median and
       RECURSE on each side.

    Args:
        points: (N, 3) array of point positions.
        rgbs: (N, 3) array of point colors.
        intrinsic_matrices: List of (3, 3) camera intrinsic matrices.
        camera_2_world_matrices: List of (4, 4) camera-to-world matrices.
        image_sizes: (M, 2) array of image sizes (height, width)
        params: PointCloudSubsamplingParams dataclass with parameters for subsampling.
    """
    pass
