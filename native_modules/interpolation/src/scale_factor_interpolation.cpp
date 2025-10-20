#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "dt_interp.h"

namespace mdi {
namespace py = pybind11;

py::array_t<FloatT> interpolate_scale_factors(const py::array_t<FloatT>& points, const py::array_t<FloatT>& scales,
                                              const py::array_t<bool>& boundary_map, size_t width, size_t height) {
    Interpolator2D interpolator(points, scales, boundary_map);

    // Interpolate on regular grid of same size as boundary map
    return interpolator.interpolate().reshape({static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width)});
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
          py::arg("boundary_map"), py::arg("width"), py::arg("height"));
}
} // namespace mdi
