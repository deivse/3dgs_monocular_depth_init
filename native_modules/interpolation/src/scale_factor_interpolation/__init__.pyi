"""
3DGS Monocular Depth Initialization Pointcloud Subsampling Module
-----------------------

.. currentmodule:: pointcloud_subsampling

.. autosummary::
    :toctree: _generate

    subsample_pointcloud
"""

import numpy


def interpolate_scale_factors(points: numpy.ndarray, scales: numpy.ndarray, boundary_map: numpy.ndarray, width: int, height: int) -> numpy.ndarray:
    """
    Interpolates scale factors at given 2D points using Delaunay triangulation.
    Args:
        points: (N, 2) array of 2D point positions where scale factors are known.
        scales: (N,) array of scale factors corresponding to the input points.
        boundary_map: (H, W) boolean array indicating borders where interpolation should not cross.
        width: Width of the output grid.
        height: Height of the output grid.

    Returns:
        (W * H,) array of interpolated scale factors at the query points.
    """
    pass
