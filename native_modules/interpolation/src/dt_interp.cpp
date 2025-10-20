#include "dt_interp.h"
#include "raster_utils.h"

#include <CGAL/Barycentric_coordinates_2/triangle_coordinates_2.h>

#include <fmt/format.h>

namespace mdi {

namespace {
    std::tuple<bool, bool, bool> check_vertex_visibility(const auto& unchecked_boundary, IntT x, IntT y,
                                                         const FVector2D& v0, const FVector2D& v1,
                                                         const FVector2D& v2) {
        auto is_visible = [&](const FVector2D& v) {
            IVector2D v_int(static_cast<IntT>(std::round(v.x())), static_cast<IntT>(std::round(v.y())));
            return raster::for_each_line_pixel(IVector2D(x, y), v_int,
                                               [&](IntT lx, IntT ly) { return !unchecked_boundary(ly, lx); });
        };
        return std::make_tuple(is_visible(v0), is_visible(v1), is_visible(v2));
    };

    std::array<FloatT, 3> compute_barycentric_coords(const std::array<FVector2D, 3>& triangle_points,
                                                     const FVector2D& p) {
        Point cgal_points[3] = {Point(triangle_points[0].x(), triangle_points[0].y()),
                                Point(triangle_points[1].x(), triangle_points[1].y()),
                                Point(triangle_points[2].x(), triangle_points[2].y())};
        Point cgal_query(p.x(), p.y());

        std::array<FloatT, 3> out;
        CGAL::Barycentric_coordinates::triangle_coordinates_2(cgal_points[0], cgal_points[1], cgal_points[2],
                                                              cgal_query, out.data());
        return out;
    }
} // namespace

Interpolator2D::Interpolator2D(const py::array_t<FloatT>& points, const py::array_t<FloatT>& values,
                               py::array_t<bool> boundary_map)
  : _triangulation(std::make_unique<Delaunay_2>()), _boundary_map(std::move(boundary_map)) {
    if (points.shape(0) != values.shape(0)) {
        throw std::invalid_argument("Points and values arrays must have the same number of elements");
    }
    if (points.shape(1) != 2) {
        throw std::invalid_argument("Points array must have shape (N, 2)");
    }
    if (values.ndim() != 1) {
        throw std::invalid_argument("Values array must be one-dimensional");
    }
    if (_boundary_map.ndim() != 2) {
        throw std::invalid_argument("Boundary map must be two-dimensional (determines output grid size)");
    }
    const auto unchecked_pts = points.unchecked<2>();
    const auto unchecked_vals = values.unchecked<1>();

    for (size_t i = 0; i < points.shape(0); ++i) {
        Vertex_handle vh = _triangulation->insert({unchecked_pts(i, 0), unchecked_pts(i, 1)});
        vh->info() = unchecked_vals(i);
    }

    const auto unchecked_boundary = _boundary_map.unchecked<2>();
    _triangle_has_boundary.resize(_triangulation->number_of_faces());
    // Iterate over each triangle
    size_t tri_index = 0;
    for (auto fit = _triangulation->finite_faces_begin(); fit != _triangulation->finite_faces_end(); ++fit) {
        _triangle_has_boundary[tri_index] = !raster::for_each_triangle_pixel(
          FVector2D(fit->vertex(0)->point().x(), fit->vertex(0)->point().y()),
          FVector2D(fit->vertex(1)->point().x(), fit->vertex(1)->point().y()),
          FVector2D(fit->vertex(2)->point().x(), fit->vertex(2)->point().y()), [&](IntT x, IntT y) {
              // Check boundary map
              if (y < 0 || y >= static_cast<IntT>(_boundary_map.shape(0)) || x < 0
                  || x >= static_cast<IntT>(_boundary_map.shape(1))) {
                  throw std::out_of_range(fmt::format("Triangle pixel ({}, {}) is out of boundary map range ({}, {})",
                                                      x, y, _boundary_map.shape(1), _boundary_map.shape(0)));
              }

              return !unchecked_boundary(y, x); // Stop if boundary pixel is found
          });
        ++tri_index;
    }
}

py::array_t<FloatT> Interpolator2D::interpolate() const {
    const auto&& [width, height] = std::tuple{_boundary_map.shape(1), _boundary_map.shape(0)};
    py::array_t<FloatT, py::array::c_style> out_array({height, width});
    auto out = out_array.mutable_unchecked<2>();

    const auto unchecked_boundary = _boundary_map.unchecked<2>();

    // Iterate over each triangle
    size_t tri_index = 0;
    for (auto tri = _triangulation->finite_faces_begin(); tri != _triangulation->finite_faces_end(); ++tri) {
        const FVector2D v0 = FVector2D(tri->vertex(0)->point().x(), tri->vertex(0)->point().y());
        const FVector2D v1 = FVector2D(tri->vertex(1)->point().x(), tri->vertex(1)->point().y());
        const FVector2D v2 = FVector2D(tri->vertex(2)->point().x(), tri->vertex(2)->point().y());

        if (!_triangle_has_boundary[tri_index]) {
            raster::for_each_triangle_pixel(v0, v1, v2, [&](IntT x, IntT y) {
                const auto barycentric_coords = compute_barycentric_coords({v0, v1, v2}, {x, y});
                out(y, x) = barycentric_coords[0] * tri->vertex(0)->info()
                            + barycentric_coords[1] * tri->vertex(1)->info()
                            + barycentric_coords[2] * tri->vertex(2)->info();
                return true;
            });
        } else {
            raster::for_each_triangle_pixel(v0, v1, v2, [&](IntT x, IntT y) {
                auto [vis0, vis1, vis2] = check_vertex_visibility(unchecked_boundary, x, y, v0, v1, v2);
                FloatT num_visible = static_cast<FloatT>(vis0 + vis1 + vis2);

                auto barycentric_coords = compute_barycentric_coords({v0, v1, v2}, {x, y});

                FloatT sum_bar_visible
                  = vis0 * barycentric_coords[0] + vis1 * barycentric_coords[1] + vis2 * barycentric_coords[2];

                if (sum_bar_visible == 0) {
                    if (num_visible == 1) {
                        // Point must lie on edge between 2 invisible vertices, third one is visible
                        // Only one will be used, but set all to 1 for simplicity
                        barycentric_coords = {FloatT(1), FloatT(1), FloatT(1)};
                        sum_bar_visible = FloatT(1);
                    } else if (num_visible == 0) {
                        // All vertices are invisible, fallback to using all vertices
                        vis0 = vis1 = vis2 = true;
                        num_visible = 3;
                        sum_bar_visible = barycentric_coords[0] + barycentric_coords[1] + barycentric_coords[2];
                    } else if (barycentric_coords[0] < 0 || barycentric_coords[1] < 0 || barycentric_coords[2] < 0) {
                        // Point is very close to boundary, so computation is inaccurate, use all vertices
                        vis0 = vis1 = vis2 = true;
                        num_visible = 3;
                        sum_bar_visible = barycentric_coords[0] + barycentric_coords[1] + barycentric_coords[2];
                        if (sum_bar_visible <= 0) {
                            const auto debug_str = fmt::format(
                              "Non-positive sum_bar_visible ({}) at pixel ({}, {}) with barycentric coords "
                              "({}, {}, {}) and visibility ({}, {}, {})",
                              sum_bar_visible, x, y, barycentric_coords[0], barycentric_coords[1],
                              barycentric_coords[2], vis0, vis1, vis2);
                            fmt::print(stderr, "{}\n", debug_str);
                            throw std::runtime_error(debug_str);
                        }
                    } else {
                        // This should not happen
                        const auto debug_str = fmt::format(
                          "No visible vertices but sum_bar_visible == 0 at pixel ({}, {}) with barycentric coords "
                          "({}, {}, {}) and visibility ({}, {}, {})",
                          x, y, barycentric_coords[0], barycentric_coords[1], barycentric_coords[2], vis0, vis1, vis2);
                        fmt::print(stderr, "{}\n", debug_str);
                        throw std::runtime_error(debug_str);
                    }
                }

                FloatT val = (barycentric_coords[0] * tri->vertex(0)->info() * vis0
                              + barycentric_coords[1] * tri->vertex(1)->info() * vis1
                              + barycentric_coords[2] * tri->vertex(2)->info() * vis2)
                             / sum_bar_visible;

                out(y, x) = val;

                // Check if NaN
                if (std::isnan(val) || !std::isfinite(val)) {
                    // print all info we have for debugging
                    const auto debug_str = fmt::format("{} encountered at pixel ({}, {}) with barycentric coords "
                                                       "({}, {}, {}) and visibility ({}, {}, {})",
                                                       val, x, y, barycentric_coords[0], barycentric_coords[1],
                                                       barycentric_coords[2], vis0, vis1, vis2);
                    fmt::print(stderr, "{}\n", debug_str);
                    throw std::runtime_error(debug_str);
                }

                return true;
            });
        }
        ++tri_index;
    }

    for (IntT x = 0; x < static_cast<IntT>(width); ++x) {
        out(height - 1, x) = interpolate(x, height - 1);
    }

    return std::move(out_array);
}

FloatT Interpolator2D::interpolate(FloatT x, FloatT y) const {
    Point query_point(x, y);
    const auto unchecked_boundary = _boundary_map.unchecked<2>();

    // Locate the point in the triangulation
    Delaunay_2::Locate_type lt;
    int li;
    Face_handle face = _triangulation->locate(query_point, lt, li);

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
        Vertex_handle v0 = face->vertex(face->ccw(li));
        Vertex_handle v1 = face->vertex(face->cw(li));

        Point p0 = v0->point();
        Point p1 = v1->point();

        // Linear interpolation along the edge
        FloatT total_dist = std::sqrt(std::pow(p1.x() - p0.x(), 2) + std::pow(p1.y() - p0.y(), 2));
        if (total_dist < 1e-15) {
            return (v0->info() + v1->info()) / FloatT(2); // Points are essentially the same
        }

        IVector2D p0_int(static_cast<IntT>(std::round(p0.x())), static_cast<IntT>(std::round(p0.y())));
        IVector2D p1_int(static_cast<IntT>(std::round(p1.x())), static_cast<IntT>(std::round(p1.y())));

        FloatT dist_to_p0 = std::sqrt(std::pow(query_point.x() - p0.x(), 2) + std::pow(query_point.y() - p0.y(), 2));

        FloatT t = dist_to_p0 / total_dist;

        FVector2D first_boundary_pixel;
        if (raster::for_each_line_pixel(p0_int, p1_int, [&first_boundary_pixel, &unchecked_boundary](IntT x, IntT y) {
                if (unchecked_boundary(y, x)) {
                    first_boundary_pixel = FVector2D(x + FloatT(0.5), y + FloatT(0.5));
                    return false; // Stop iteration
                }
                return true;
            })) {
            // No boundary pixel found between the two vertices - do linear interpolation
            return t * v0->info() + (1 - t) * v1->info();
        }
        // Vertices separated by boundary
        FloatT boundary_dist_to_p0
          = std::sqrt((first_boundary_pixel.x() - p0.x()) * (first_boundary_pixel.x() - p0.x())
                      + (first_boundary_pixel.y() - p0.y()) * (first_boundary_pixel.y() - p0.y()));
        FloatT t_boundary = boundary_dist_to_p0 / total_dist;
        return t < t_boundary ? v0->info() : v1->info();
    }

    if (lt == Delaunay_2::FACE) {
        // Query point lies inside a triangle - use barycentric interpolation
        std::array<Point, 3> triangle_points
          = {face->vertex(0)->point(), face->vertex(1)->point(), face->vertex(2)->point()};
        FVector2D v0 = FVector2D(triangle_points[0].x(), triangle_points[0].y());
        FVector2D v1 = FVector2D(triangle_points[1].x(), triangle_points[1].y());
        FVector2D v2 = FVector2D(triangle_points[2].x(), triangle_points[2].y());

        const auto barycentric = compute_barycentric_coords({v0, v1, v2}, {x, y});
        auto [vis0, vis1, vis2] = check_vertex_visibility(unchecked_boundary, x, y, v0, v1, v2);

        FloatT num_visible = static_cast<FloatT>(vis0 + vis1 + vis2);
        if (num_visible == 0) {
            vis0 = vis1 = vis2 = true;
            num_visible = 3;
        }

        FloatT sum_bar_visible = vis0 * barycentric[0] + vis1 * barycentric[1] + vis2 * barycentric[2];
        FloatT result
          = (barycentric[0] * face->vertex(0)->info() * vis0 + barycentric[1] * face->vertex(1)->info() * vis1
             + barycentric[2] * face->vertex(2)->info() * vis2)
            / sum_bar_visible;

        return result;
    }

    // Should not reach here
    throw std::runtime_error("Unexpected location type when querying triangulation");
}

} // namespace mdi
