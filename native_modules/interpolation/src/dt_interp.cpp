#include "dt_interp.h"

namespace mdi {

DelaunayInterpolator2D::DelaunayInterpolator2D(const py::array_t<FloatT>& points, const py::array_t<FloatT>& values)
  : m_triangulation(std::make_unique<Delaunay_2>()) {
    if (points.shape(0) != values.shape(0)) {
        throw std::invalid_argument("Points and values arrays must have the same number of elements");
    }
    if (points.shape(1) != 2) {
        throw std::invalid_argument("Points array must have shape (N, 2)");
    }
    if (values.ndim() != 1) {
        throw std::invalid_argument("Values array must be one-dimensional");
    }

    const auto unchecked_pts = points.unchecked<2>();
    const auto unchecked_vals = values.unchecked<1>();

    for (size_t i = 0; i < points.shape(0); ++i) {
        Vertex_handle vh = m_triangulation->insert({unchecked_pts(i, 0), unchecked_pts(i, 1)});
        vh->info() = unchecked_vals(i);
    }
}

py::array_t<FloatT> DelaunayInterpolator2D::interpolate_regular_grid(FloatT x_min, FloatT x_max, int nx, FloatT y_min,
                                                                     FloatT y_max, int ny) const {
    common::py_output_array<FloatT, 1> grid(ny * nx);

    FloatT dx = (nx > 1) ? (x_max - x_min) / (nx - 1) : 0.0;
    FloatT dy = (ny > 1) ? (y_max - y_min) / (ny - 1) : 0.0;

    for (int j = 0; j < ny; ++j) {
        FloatT y = y_min + j * dy;
        for (int i = 0; i < nx; ++i) {
            FloatT x = x_min + i * dx;
            grid.push_back(interpolate(x, y));
        }
    }

    return std::move(grid).finalize();
}

std::vector<FloatT> DelaunayInterpolator2D::compute_barycentric_coords(const std::array<Point, 3>& triangle_points,
                                                                       const Point& x) const {
    const Point& p0 = triangle_points[0];
    const Point& p1 = triangle_points[1];
    const Point& p2 = triangle_points[2];

    // Compute vectors
    FloatT v0x = p2.x() - p0.x();
    FloatT v0y = p2.y() - p0.y();
    FloatT v1x = p1.x() - p0.x();
    FloatT v1y = p1.y() - p0.y();
    FloatT v2x = x.x() - p0.x();
    FloatT v2y = x.y() - p0.y();

    // Compute dot products
    FloatT dot00 = v0x * v0x + v0y * v0y;
    FloatT dot01 = v0x * v1x + v0y * v1y;
    FloatT dot02 = v0x * v2x + v0y * v2y;
    FloatT dot11 = v1x * v1x + v1y * v1y;
    FloatT dot12 = v1x * v2x + v1y * v2y;

    // Compute barycentric coordinates
    FloatT inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    FloatT u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    FloatT v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    FloatT w = 1.0 - u - v;

    return {w, v, u}; // weights for p0, p1, p2
}

FloatT DelaunayInterpolator2D::interpolate(FloatT x, FloatT y) const {
    Point query_point(x, y);

    // Locate the point in the triangulation
    Delaunay_2::Locate_type lt;
    int li;
    Face_handle face = m_triangulation->locate(query_point, lt, li);

    // Handle different location results
    if (face == nullptr || lt == Delaunay_2::OUTSIDE_CONVEX_HULL || lt == Delaunay_2::OUTSIDE_AFFINE_HULL) {
        // Point is outside the convex hull, this should be pre-handled by adding extra points at corners
        throw std::runtime_error("Interpolation point is outside the convex hull");
    }

    if (lt == Delaunay_2::VERTEX) {
        // Query point coincides with a vertex
        return face->vertex(li)->info();
    }

    if (lt == Delaunay_2::EDGE) {
        // Query point lies on an edge - interpolate between the two vertices
        Vertex_handle v1 = face->vertex(face->ccw(li));
        Vertex_handle v2 = face->vertex(face->cw(li));

        Point p1 = v1->point();
        Point p2 = v2->point();

        // Linear interpolation along the edge
        FloatT total_dist = std::sqrt((p2.x() - p1.x()) * (p2.x() - p1.x()) + (p2.y() - p1.y()) * (p2.y() - p1.y()));
        if (total_dist < 1e-15) {
            return v1->info(); // Points are essentially the same
        }

        FloatT dist_to_p1 = std::sqrt((query_point.x() - p1.x()) * (query_point.x() - p1.x())
                                      + (query_point.y() - p1.y()) * (query_point.y() - p1.y()));

        FloatT t = dist_to_p1 / total_dist;
        return (1.0 - t) * v1->info() + t * v2->info();
    }

    if (lt == Delaunay_2::FACE) {
        // Query point lies inside a triangle - use barycentric interpolation
        std::array<Point, 3> triangle_points
          = {face->vertex(0)->point(), face->vertex(1)->point(), face->vertex(2)->point()};

        std::vector<FloatT> barycentric = compute_barycentric_coords(triangle_points, query_point);

        FloatT result = 0.0;
        for (int i = 0; i < 3; ++i) {
            result += barycentric[i] * face->vertex(i)->info();
        }

        return result;
    }

    // Should not reach here
    throw std::runtime_error("Unexpected location type when querying triangulation");
}

} // namespace mdi
