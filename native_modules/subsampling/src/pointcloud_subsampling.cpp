#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

py::array_t<double> subsample_pointcloud(py::array_t<double> points) {
    return points;
}


PYBIND11_MODULE(_pointcloud_subsampling, m) {
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
