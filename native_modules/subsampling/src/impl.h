#ifndef NATIVE_MODULES_SUBSAMPLING_SRC_IMPL
#define NATIVE_MODULES_SUBSAMPLING_SRC_IMPL

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>

#include "defines.h"

namespace py = pybind11;

namespace mdi::pointcloud {

struct PointLocation
{
    FloatT x;
    FloatT y;
    FloatT z;
    FloatT _pad;
};

struct PointCloud
{
    py::array_t<FloatT> positions;
    py::array_t<FloatT> rgbs;
};

std::vector<FloatT> compute_minimal_gaussian_extents(const py::array_t<FloatT>& points,
                                                     const std::vector<FMatrix3D>& intrinsic_matrices,
                                                     const std::vector<FMatrix3x4>& camera_2_world_matrices,
                                                     const py::array_t<IntT>& image_sizes,
                                                     const std::vector<std::pair<IntT, IntT>>& points_to_cam_slices);

PointCloud subsample_pointcloud_impl(const PointCloud& pointcloud, const std::vector<FloatT>& min_gaussian_extents,
                                     FloatT min_extent_mult);

} // namespace mdi::pointcloud

#endif /* NATIVE_MODULES_SUBSAMPLING_SRC_IMPL */
