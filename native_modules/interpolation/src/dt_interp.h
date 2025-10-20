#ifndef NATIVE_MODULES_INTERPOLATION_SRC_DT_INTERP
#define NATIVE_MODULES_INTERPOLATION_SRC_DT_INTERP

#include <vector>
#include <array>
#include <memory>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <boost/dynamic_bitset.hpp>

#include <common/py_output_array.h>
#include "defines.h"

namespace mdi {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Vertex_base = CGAL::Triangulation_vertex_base_with_info_2<FloatT, Kernel>;
using TDS = CGAL::Triangulation_data_structure_2<Vertex_base>;
using Delaunay_2 = CGAL::Delaunay_triangulation_2<Kernel, TDS>;
using Point = Kernel::Point_2;
using Vertex_handle = Delaunay_2::Vertex_handle;
using Face_handle = Delaunay_2::Face_handle;
class Interpolator2D
{
private:
    std::unique_ptr<Delaunay_2> _triangulation;
    py::array_t<bool> _boundary_map;
    boost::dynamic_bitset<> _triangle_has_boundary;

public:
    Interpolator2D(const py::array_t<FloatT>& points, const py::array_t<FloatT>& values,
                   py::array_t<bool> boundary_map);

    py::array_t<FloatT> interpolate() const;

    // Get number of points in the triangulation
    size_t size() const { return _triangulation->number_of_vertices(); }

    // Check if triangulation is valid (has at least 3 points)
    bool is_valid() const { return _triangulation->number_of_vertices() >= 3; }

private:
    // Interpolate at a given point
    FloatT interpolate(FloatT x, FloatT y) const;
};
} // namespace mdi

#endif /* NATIVE_MODULES_INTERPOLATION_SRC_DT_INTERP */
