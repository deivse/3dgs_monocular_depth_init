#ifndef DT_INTERP_H
#define DT_INTERP_H

#include <vector>
#include <array>
#include <memory>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <common/py_output_array.h>
#include "defines.h"

namespace mdi {
class DelaunayInterpolator2D
{
private:
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Vertex_base = CGAL::Triangulation_vertex_base_with_info_2<FloatT, Kernel>;
    using TDS = CGAL::Triangulation_data_structure_2<Vertex_base>;
    using Delaunay_2 = CGAL::Delaunay_triangulation_2<Kernel, TDS>;
    using Point = Kernel::Point_2;
    using Vertex_handle = Delaunay_2::Vertex_handle;
    using Face_handle = Delaunay_2::Face_handle;

    std::unique_ptr<Delaunay_2> m_triangulation;

public:
    DelaunayInterpolator2D(const py::array_t<FloatT>& points, const py::array_t<FloatT>& values);

    py::array_t<FloatT> interpolate_regular_grid(FloatT x_min, FloatT x_max, int nx, FloatT y_min, FloatT y_max,
                                                 int ny) const;

    // Get number of points in the triangulation
    size_t size() const { return m_triangulation->number_of_vertices(); }

    // Check if triangulation is valid (has at least 3 points)
    bool is_valid() const { return m_triangulation->number_of_vertices() >= 3; }

private:
    // Compute barycentric coordinates for point x in triangle defined by 3 vertices
    std::vector<FloatT> compute_barycentric_coords(const std::array<Point, 3>& triangle_points, const Point& x) const;

    // Interpolate at a given point
    FloatT interpolate(FloatT x, FloatT y) const;
};
} // namespace mdi

#endif // DT_INTERP_H
