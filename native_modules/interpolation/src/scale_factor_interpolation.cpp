#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "dt_interp.h"

namespace mdi {
namespace py = pybind11;

py::array_t<FloatT> interpolate_scale_factors(const py::array_t<FloatT>& points, const py::array_t<FloatT>& scales,
                                              size_t width, size_t height) {
    // Create interpolator
    DelaunayInterpolator2D interpolator(points, scales);

    // Interpolate on regular grid
    return interpolator
      .interpolate_regular_grid(0.0f, static_cast<FloatT>(width - 1), static_cast<int>(width), 0.0f,
                                static_cast<FloatT>(height - 1), static_cast<int>(height))
      .reshape({static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width)});
}

PYBIND11_MODULE(_scale_factor_interpolation, m) {
    m.doc() = R"pbdoc(
        3DGS Monocular Depth Initialization Scale Factor Interpolation Module
        -----------------------

        .. currentmodule:: scale_factor_interpolation

        .. autosummary::
           :toctree: _generate

           interpolate_scale_factors
    )pbdoc";
    m.def("interpolate_scale_factors", &interpolate_scale_factors, py::arg("points"), py::arg("scales"),
          py::arg("width"), py::arg("height"));
}
} // namespace mdi
