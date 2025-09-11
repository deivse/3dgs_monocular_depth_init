#include "impl.h"
#include "geometry.h"
#include "stack.h"

#include <Eigen/Dense>
#include <ranges>
#include <algorithm>
#include <span>

namespace mdi::pointcloud {

std::optional<FloatT> get_point_depth_in_camera(const FVector3D& point, const FMatrix3D& K,
                                                const FMatrix3x4& proj_matrix, const Eigen::Vector2i& image_size) {
    const auto homogeneous_point = point.homogeneous();
    const auto projected_point = proj_matrix * homogeneous_point;
    const FloatT depth = projected_point.z();

    if (depth <= 0) {
        return std::nullopt;
    }

    const auto u = projected_point.x() / depth;
    const auto v = projected_point.y() / depth;

    if (u < 0 || u >= image_size.x() || v < 0 || v >= image_size.y()) {
        return std::nullopt;
    }

    return depth;
}

std::vector<FloatT> compute_minimal_gaussian_extents(const py::array_t<FloatT>& points,
                                                     const std::vector<FMatrix3D>& intrinsic_matrices,
                                                     const std::vector<FMatrix3x4>& proj_matrices,
                                                     const py::array_t<IntT>& image_sizes,
                                                     const std::vector<std::pair<IntT, IntT>>& points_to_cam_slices) {
    std::vector<FloatT> min_gaussian_extents(points.shape(0), std::numeric_limits<FloatT>::max());

    const auto num_cameras = intrinsic_matrices.size();

    const auto points_unchecked = points.unchecked<2>();
    const auto image_sizes_unchecked = image_sizes.unchecked<2>();

    for (ssize_t i = 0; i < points.shape(0); ++i) { // TODO: thread pool cuz this is impossibly slow in debug
        const Eigen::Matrix<FloatT, 3, 1> point{points_unchecked(i, 0), points_unchecked(i, 1), points_unchecked(i, 2)};

        for (size_t cam = 0; cam < num_cameras; ++cam) {
            const FMatrix3D& intrinsic_matrix = intrinsic_matrices[cam];
            const FMatrix3x4& proj_matrix = proj_matrices[cam];
            const Eigen::Vector2i image_size
              = Eigen::Vector2i(image_sizes_unchecked(cam, 0), image_sizes_unchecked(cam, 1));

            const auto depth_opt = get_point_depth_in_camera(point, intrinsic_matrix, proj_matrix, image_size);
            if (!depth_opt.has_value()) {
                continue;
            }

            const FloatT depth = *depth_opt;

            const FloatT fx = intrinsic_matrix(0, 0);
            const FloatT fy = intrinsic_matrix(1, 1);

            const FloatT f = std::min(fx, fy); // Conservative choice

            const FloatT world_space_sampling_interval = depth / f;

            min_gaussian_extents[i]
              = std::min<FloatT>(min_gaussian_extents[i], static_cast<FloatT>(2.0) * world_space_sampling_interval);
        }
    }

    for (auto& extent : min_gaussian_extents) {
        if (extent == std::numeric_limits<FloatT>::max()) {
            extent = static_cast<FloatT>(-1.0); // No valid observation, set extent to -1
        }
    }

    return min_gaussian_extents;
}

PointCloud subsample_pointcloud_impl(const PointCloud& pointcloud, const std::vector<FloatT>& min_gaussian_extents,
                                     FloatT min_extent_mult) {
    // Implementation of the subsampling algorithm
    PointCloud subsampled_pointcloud{
      py::array_t<FloatT, py::array::c_style>({pointcloud.positions.shape(0), py::ssize_t(3)}),
      py::array_t<FloatT, py::array::c_style>({pointcloud.positions.shape(0), py::ssize_t(3)})};

    const auto unchecked_positions = pointcloud.positions.unchecked<2>();
    const auto unchecked_rgbs = pointcloud.rgbs.unchecked<2>();
    size_t num_output_points = 0;
    auto unchecked_out_positions = subsampled_pointcloud.positions.mutable_unchecked<2>();
    auto unchecked_out_rgbs = subsampled_pointcloud.rgbs.mutable_unchecked<2>();

    std::vector<FVector3D> positions;
    positions.reserve(pointcloud.positions.shape(0));
    for (ssize_t i = 0; i < pointcloud.positions.shape(0); ++i) {
        positions.emplace_back(unchecked_positions(i, 0), unchecked_positions(i, 1), unchecked_positions(i, 2));
    }

    std::vector<uint32_t> indices(positions.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Similar to kd-tree construction except no data structure is stored in memory, in leaves we immediately output
    // merged points.
    struct StackFrame
    {
        std::span<uint32_t> indices;
        geom::BoundingBox box;
        geom::Axis axis;
    };

    Stack<StackFrame> stack{};
    stack.emplace(StackFrame{std::span(indices), geom::BoundingBox::from_points(positions), geom::Axis::X});

    const auto output_point = [&](const FVector3D& pos, const FVector3D& rgb) {
        *reinterpret_cast<FVector3D*>(unchecked_out_positions.mutable_data(num_output_points, 0)) = pos;
        *reinterpret_cast<FVector3D*>(unchecked_out_rgbs.mutable_data(num_output_points, 0)) = rgb;
        num_output_points++;
    };

    while (!stack.empty()) {
        const auto frame = stack.pop();
        if (frame.indices.size() == 0) {
            continue;
        }

        FloatT avg_min_ext = 0;
        for (auto ix : frame.indices) {
            const auto extent = min_gaussian_extents[ix];
            avg_min_ext += extent;
        }
        avg_min_ext /= static_cast<FloatT>(frame.indices.size());

        const auto bb_min_len = frame.box.diagonal().minCoeff();
        if (bb_min_len <= min_extent_mult * avg_min_ext) {
            // Merge points
            FVector3D merged_pos = FVector3D::Zero();
            FVector3D merged_rgb = FVector3D::Zero();
            for (auto ix : frame.indices) {
                merged_pos += positions[ix];
                merged_rgb += *reinterpret_cast<const FVector3D*>(unchecked_rgbs.data(ix, 0));
            }
            merged_pos /= static_cast<FloatT>(frame.indices.size());
            merged_rgb /= static_cast<FloatT>(frame.indices.size());
            output_point(merged_pos, merged_rgb);
            continue;
        }
        if (frame.indices.size() <= 2) {
            // Output all points
            for (auto ix : frame.indices) {
                output_point(positions[ix], *reinterpret_cast<const FVector3D*>(unchecked_rgbs.data(ix, 0)));
            }
            continue;
        }

        // Split
        geom::Axis next_axis = geom::axis_from_int((geom::axis_to_int(frame.axis) + 1) % 3);

        const auto axis_idx = static_cast<size_t>(frame.axis);
        FloatT split_value = 0.5 * (frame.box.min[axis_idx] + frame.box.max[axis_idx]);
        auto mid_it = std::partition(frame.indices.begin(), frame.indices.end(),
                                     [&](uint32_t ix) { return positions[ix][axis_idx] < split_value; });

        if (mid_it == frame.indices.begin() || mid_it == frame.indices.end()) {
            mid_it = frame.indices.begin() + frame.indices.size() / 2;
            std::ranges::nth_element(frame.indices, mid_it, [&](uint32_t a, uint32_t b) {
                return positions[a][axis_idx] < positions[b][axis_idx];
            });
            split_value = (positions[*(mid_it - 1)][axis_idx] + positions[*mid_it][axis_idx]) / FloatT(2);
        }
        const auto left_indices = std::span(frame.indices.begin(), mid_it);
        const auto right_indices = std::span(mid_it, frame.indices.end());
        const auto [left_box, right_box] = frame.box.split(frame.axis, split_value);

        stack.emplace(StackFrame{right_indices, right_box, next_axis});
        stack.emplace(StackFrame{left_indices, left_box, next_axis});
    }

    subsampled_pointcloud.positions.resize({py::ssize_t(num_output_points), py::ssize_t(3)});
    subsampled_pointcloud.rgbs.resize({py::ssize_t(num_output_points), py::ssize_t(3)});

    return subsampled_pointcloud;
}
} // namespace mdi::pointcloud
