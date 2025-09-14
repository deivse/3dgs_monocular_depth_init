#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>

#include "py_output_array.h"
#include "impl.h"

namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

constexpr auto py_array_packed_row_major = py::array::c_style;

namespace mdi::pointcloud {
using input_float_array_t = py::array_t<FloatT, py_array_packed_row_major | py::array::forcecast>;
using input_int_array_t = py::array_t<IntT, py_array_packed_row_major | py::array::forcecast>;

std::tuple<py::array_t<FloatT>, py::array_t<FloatT>, py::array_t<FloatT>, py::array_t<FloatT>, py::array_t<FloatT>>
  subsample_pointcloud(input_float_array_t points, input_float_array_t rgbs,
                       const std::vector<FMatrix3D>& intrinsic_matrices,
                       const std::vector<FMatrix3x4>& camera_2_world_matrices, input_int_array_t image_sizes,
                       const std::vector<std::pair<IntT, IntT>>& points_to_cam_slices, FloatT min_extent_mult = 1.0f) {
    constexpr auto PointsDataDim = 3;
    constexpr auto RgbsDataDim = 3;
    constexpr auto ImageSizesDataDim = 2;

    if (points.ndim() != 2 || points.shape(1) != PointsDataDim) {
        throw py::value_error("Input points array must have shape (N, 3)");
    }

    if (rgbs.ndim() != 2 || rgbs.shape(1) != RgbsDataDim) {
        throw py::value_error("Input rgbs array must have shape (N, 3)");
    }

    if (image_sizes.ndim() != 2 || image_sizes.shape(1) != ImageSizesDataDim) {
        throw py::value_error("Input image_sizes array must have shape (N, 2)");
    }

    if (rgbs.shape(0) != points.shape(0)) {
        throw py::value_error("Number of points must match number of rgbs.");
    }

    if (intrinsic_matrices.size() != camera_2_world_matrices.size()
        || intrinsic_matrices.size() != image_sizes.shape(0)) {
        throw py::value_error(
          "Number of intrinsic_matrices must match number of camera_2_world_matrices and image_sizes.");
    }

    const auto min_gaussian_extents = compute_minimal_gaussian_extents(
      points, intrinsic_matrices, camera_2_world_matrices, image_sizes, points_to_cam_slices);

    auto&& [subsampled, debug_out]
      = subsample_pointcloud_impl(PointCloud{points, rgbs}, min_gaussian_extents, min_extent_mult);

    py::array_t<FloatT, py::array::c_style> out_extents(min_gaussian_extents.size());
    std::memcpy(out_extents.mutable_data(), min_gaussian_extents.data(), min_gaussian_extents.size() * sizeof(FloatT));

    return {std::move(subsampled.positions), std::move(subsampled.rgbs), std::move(out_extents), std::move(debug_out.positions), std::move(debug_out.rgbs)};
}

PYBIND11_MODULE(_pointcloud_subsampling, m, py::mod_gil_not_used()) {
    m.doc() = R"pbdoc(
        3DGS Monocular Depth Initialization Pointcloud Subsampling Module
        -----------------------

        .. currentmodule:: pointcloud_subsampling

        .. autosummary::
           :toctree: _generate

           subsample_pointcloud
    )pbdoc";

    m.def("subsample_pointcloud", &subsample_pointcloud);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

} // namespace mdi::pointcloud
