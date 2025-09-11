"""
3DGS Monocular Depth Initialization Pointcloud Subsampling Module
-----------------------

.. currentmodule:: pointcloud_subsampling

.. autosummary::
    :toctree: _generate

    subsample_pointcloud
"""

import numpy

def subsample_pointcloud(
    points: numpy.ndarray,
    rgbs: numpy.ndarray,
    intrinsic_matrices: list[numpy.ndarray],
    camera_2_world_matrices: list[numpy.ndarray],
    image_sizes: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    pass
