#include "impl.h"
#include "debug.h"
#include "geometry.h"
#include "stack.h"
#include "stat_accumulator.h"

#include <Eigen/Dense>
#include <iostream>
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

// TODO: refactor and collect stats about:
// - sizes of BB of merged points
// - distribution of min_gaussian_extents of merged points
namespace {
    struct SubsamplePointcloudStats
    {
        StatAccumulator bbox_aspect_ratio;
        StatAccumulator bbox_min_size;
        StatAccumulator bbox_max_size;
        StatAccumulator bbox_biggest_dim_above_extent;
        StatAccumulator merged_min_gaussian_extent;
        StatAccumulator merged_num_points_in_leaf;
        std::chrono::high_resolution_clock::time_point start_time, end_time;
        std::chrono::duration<double> duration;
        size_t total_splits = 0;
        size_t total_fallback_splits = 0;
        size_t num_merged_points = 0;
        size_t num_unmerged_points = 0;

        void add_merged(const geom::BoundingBox& box, FloatT min_gaussian_extent, size_t num_points_in_leaf) {
            const auto diag = box.diagonal();
            const auto aspect_ratio = diag.maxCoeff() / diag.minCoeff();
            bbox_aspect_ratio.add(aspect_ratio);
            bbox_min_size.add(diag.minCoeff());
            bbox_max_size.add(diag.maxCoeff());
            bbox_biggest_dim_above_extent.add(std::max(FloatT(0), diag.maxCoeff() - min_gaussian_extent));
            merged_min_gaussian_extent.add(min_gaussian_extent);
            merged_num_points_in_leaf.add(static_cast<FloatT>(num_points_in_leaf));
            num_merged_points++;
        }

        void add_unmerged() { num_unmerged_points++; }

        void start_timer() { start_time = std::chrono::high_resolution_clock::now(); }
        void stop_timer() {
            end_time = std::chrono::high_resolution_clock::now();
            duration = end_time - start_time;
        }

        void add_split(bool fallback_used) {
            total_splits++;
            total_fallback_splits += fallback_used;
        }

        void print() const {
            std::cout << "SubsamplePointcloudStats:\n";
            std::cout << "  Merged points: " << num_merged_points << "\n";
            std::cout << "  Unmerged points: " << num_unmerged_points << "\n";
            std::cout << "  Time taken: " << duration.count() << " seconds\n";
            if (num_merged_points > 0) {
                std::cout << "  BBox aspect ratio: mean=" << bbox_aspect_ratio.mean()
                          << ", stddev=" << bbox_aspect_ratio.stddev() << ", min=" << bbox_aspect_ratio.min
                          << ", max=" << bbox_aspect_ratio.max << "\n";
                std::cout << "  BBox min size: mean=" << bbox_min_size.mean() << ", stddev=" << bbox_min_size.stddev()
                          << ", min=" << bbox_min_size.min << ", max=" << bbox_min_size.max << "\n";
                std::cout << "  BBox max size: mean=" << bbox_max_size.mean() << ", stddev=" << bbox_max_size.stddev()
                          << ", min=" << bbox_max_size.min << ", max=" << bbox_max_size.max << "\n";
                std::cout << "  Merged min gaussian extent: mean=" << merged_min_gaussian_extent.mean()
                          << ", stddev=" << merged_min_gaussian_extent.stddev()
                          << ", min=" << merged_min_gaussian_extent.min << ", max=" << merged_min_gaussian_extent.max
                          << "\n";
                std::cout << "  Merged num points in leaf: mean=" << merged_num_points_in_leaf.mean()
                          << ", stddev=" << merged_num_points_in_leaf.stddev()
                          << ", min=" << merged_num_points_in_leaf.min << ", max=" << merged_num_points_in_leaf.max
                          << "\n";
                std::cout << "  BBox biggest dim above extent: mean=" << bbox_biggest_dim_above_extent.mean()
                          << ", stddev=" << bbox_biggest_dim_above_extent.stddev()
                          << ", min=" << bbox_biggest_dim_above_extent.min
                          << ", max=" << bbox_biggest_dim_above_extent.max << "\n";
                std::cout << "  Total splits: " << total_splits
                          << ", of which fallback strat used in: " << total_fallback_splits << "\n";
            }
        }
    };

    using IndexT = uint32_t;

    std::vector<IndexT> make_index_sequence(size_t size) {
        std::vector<IndexT> indices(size);
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }

    std::vector<FVector3D> array_to_vector(const py::array_t<FloatT>& arr) {
        std::vector<FVector3D> vec;
        vec.reserve(arr.shape(0));
        const auto unchecked = arr.unchecked<2>();
        for (ssize_t i = 0; i < arr.shape(0); ++i) {
            vec.emplace_back(unchecked(i, 0), unchecked(i, 1), unchecked(i, 2));
        }
        return vec;
    }

    struct StackFrame
    {
        std::vector<IndexT> indices;
        geom::BoundingBox box;
        geom::Axis prev_axis;
    };

    FloatT compute_average_min_gaussian_extent(const std::vector<IndexT>& indices,
                                               const std::vector<FloatT>& min_gaussian_extents) {
        FloatT avg = 0;
        for (auto ix : indices) {
            avg += min_gaussian_extents[ix];
        }
        return avg / static_cast<FloatT>(indices.size());
    }

    // Split strategies

    struct SplitResult
    {
        FloatT split_value;
        std::vector<IndexT> indices_left;
        std::vector<IndexT> indices_right;
    };

    SplitResult equal_partitions(StackFrame& frame, size_t axis_idx, const std::vector<FVector3D>& positions,
                                 FloatT /*avg_min_extent*/) {
        auto mid_it = frame.indices.begin() + frame.indices.size() / 2;
        std::ranges::nth_element(frame.indices, mid_it,
                                 [&](IndexT a, IndexT b) { return positions[a][axis_idx] < positions[b][axis_idx]; });
        FloatT split_value = (positions[*(mid_it - 1)][axis_idx] + positions[*mid_it][axis_idx]) / FloatT(2);

        std::vector<IndexT> indices_left(frame.indices.begin(), mid_it);
        std::vector<IndexT> indices_right(mid_it, frame.indices.end());
        return {split_value, std::move(indices_left), std::move(indices_right)};
    }

    SplitResult max_gap(StackFrame& frame, size_t axis_idx, const std::vector<FVector3D>& positions,
                        FloatT avg_min_extent) {
        FloatT dist_max = std::numeric_limits<FloatT>::lowest();
        FloatT mean_dist = 0;
        size_t split_idx = 0;
        std::vector<uint32_t> indices_left, indices_right;
        std::ranges::sort(frame.indices,
                          [&](uint32_t a, uint32_t b) { return positions[a][axis_idx] < positions[b][axis_idx]; });

        for (size_t i = frame.indices.size() / 4; i < frame.indices.size() - frame.indices.size() / 4; ++i) {
            FloatT dist_to_prev
              = std::abs(positions[frame.indices[i - 1]][axis_idx] - positions[frame.indices[i]][axis_idx]);
            mean_dist += dist_to_prev;
            if (dist_to_prev > dist_max) {
                dist_max = dist_to_prev;
                split_idx = i;
            }
        }
        mean_dist /= static_cast<FloatT>(frame.indices.size() - 1);
        if (dist_max < 2.0f * mean_dist) {
            // Bad split, use median
            split_idx = frame.indices.size() / 2;
        }
        FloatT split_value
          = (positions[frame.indices[split_idx - 1]][axis_idx] + positions[frame.indices[split_idx]][axis_idx])
            / FloatT(2);

        indices_left = std::vector<uint32_t>(frame.indices.begin(), frame.indices.begin() + split_idx);
        indices_right = std::vector<uint32_t>(frame.indices.begin() + split_idx, frame.indices.end());

        return {split_value, std::move(indices_left), std::move(indices_right)};
    }

    SplitResult spatial_median(StackFrame& frame, size_t axis_idx, const std::vector<FVector3D>& positions,
                               FloatT avg_min_extent) {
        FloatT split_value = (frame.box.min[axis_idx] + frame.box.max[axis_idx]) / FloatT(2);
        std::vector<IndexT> indices_left, indices_right;
        for (auto ix : frame.indices) {
            if (positions[ix][axis_idx] < split_value) {
                indices_left.push_back(ix);
            } else {
                indices_right.push_back(ix);
            }
        }

        return {split_value, std::move(indices_left), std::move(indices_right)};
    }

    SplitResult first_spatial_median_and_then_nth_element(StackFrame& frame, size_t axis_idx,
                                                          const std::vector<FVector3D>& positions,
                                                          FloatT avg_min_extent) {
        if (frame.box.diagonal().minCoeff() >= 20 * avg_min_extent) {
            return spatial_median(frame, axis_idx, positions, avg_min_extent);
        } else {
            return equal_partitions(frame, axis_idx, positions, avg_min_extent);
        }
    }

} // namespace

std::pair<PointCloud, PointCloud> subsample_pointcloud_impl(const PointCloud& pointcloud,
                                                            const std::vector<FloatT>& min_gaussian_extents,
                                                            FloatT min_extent_mult) {
    SubsamplePointcloudStats stats{};
    stats.start_timer();

    // Implementation of the subsampling algorithm
    PointCloud subsampled_pointcloud{
      py::array_t<FloatT, py::array::c_style>({pointcloud.positions.shape(0), py::ssize_t(3)}),
      py::array_t<FloatT, py::array::c_style>({pointcloud.positions.shape(0), py::ssize_t(3)})};
    PointCloud debug_out{py::array_t<FloatT, py::array::c_style>({pointcloud.positions.shape(0) * 2, py::ssize_t(3)}),
                         py::array_t<FloatT, py::array::c_style>({pointcloud.positions.shape(0) * 2, py::ssize_t(3)})};

    auto [unchecked_debug_positions, unchecked_debug_rgbs] = debug_out.get_mut_unchecked();
    const auto [unchecked_positions, unchecked_rgbs] = pointcloud.get_unchecked();
    auto [unchecked_out_positions, unchecked_out_rgbs] = subsampled_pointcloud.get_mut_unchecked();

    size_t num_output_points = 0;
    const auto output_point = [&](const FVector3D& pos, const FVector3D& rgb) {
        *reinterpret_cast<FVector3D*>(unchecked_out_positions.mutable_data(num_output_points, 0)) = pos;
        *reinterpret_cast<FVector3D*>(unchecked_out_rgbs.mutable_data(num_output_points, 0)) = rgb;
        num_output_points++;
    };

    size_t debug_out_idx = 0;
    size_t debug_group_idx = 0;
    const auto output_debug = [&](const FVector3D& dbg, const FVector3D& color) {
        *reinterpret_cast<FVector3D*>(unchecked_debug_positions.mutable_data(debug_out_idx, 0)) = dbg;
        *reinterpret_cast<FVector3D*>(unchecked_debug_rgbs.mutable_data(debug_out_idx, 0)) = color;
        debug_out_idx++;
    };

    Stack<StackFrame> stack{};

    const auto positions = array_to_vector(pointcloud.positions);
    const auto tightBB = geom::BoundingBox::from_points(positions);
    const auto squareBB
      = geom::BoundingBox{tightBB.min, tightBB.min + FVector3D::Ones() * tightBB.diagonal().maxCoeff()};
    stack.emplace(StackFrame{make_index_sequence(positions.size()), squareBB, geom::Axis::X});

    while (!stack.empty()) {
        auto frame = stack.pop();
        if (frame.indices.size() == 0) {
            continue;
            // throw std::runtime_error("Empty frame in stack, this should never happen.");
        }

        FloatT avg_min_ext = compute_average_min_gaussian_extent(frame.indices, min_gaussian_extents);

        const auto box_aspect_ratio = frame.box.diagonal().maxCoeff() / frame.box.diagonal().minCoeff();
        const auto threshold = frame.box.diagonal().minCoeff();
        if (box_aspect_ratio <= 2 && threshold <= min_extent_mult * avg_min_ext) {
            // Merge points
            FVector3D merged_pos = FVector3D::Zero();
            FVector3D merged_rgb = FVector3D::Zero();
            for (auto ix : frame.indices) {
                merged_pos += positions[ix];
                merged_rgb += *reinterpret_cast<const FVector3D*>(unchecked_rgbs.data(ix, 0));
                if (frame.indices.size() > 1) {
                    output_debug(positions[ix], mdi::debug::get_debug_color_muted(debug_group_idx));
                }
            }
            merged_pos /= static_cast<FloatT>(frame.indices.size());
            merged_rgb /= static_cast<FloatT>(frame.indices.size());

            stats.add_merged(frame.box, min_extent_mult * avg_min_ext, frame.indices.size());
            output_point(merged_pos, merged_rgb);

            if (frame.indices.size() > 1) {
                output_debug(merged_pos, mdi::debug::get_debug_color(debug_group_idx++));
            }

            continue;
        }
        if (frame.indices.size() <= 2) {
            // Output all points
            for (auto ix : frame.indices) {
                stats.add_unmerged();
                output_point(positions[ix], *reinterpret_cast<const FVector3D*>(unchecked_rgbs.data(ix, 0)));
            }
            continue;
        }

        const auto axis_idx = (geom::axis_to_int(frame.prev_axis) + 1) % geom::NumAxes;
        const auto axis = geom::axis_from_int(axis_idx);

        constexpr auto split_fn = &spatial_median;

        auto [split_value, indices_left, indices_right] = split_fn(frame, axis_idx, positions, avg_min_ext);

        bool fallback_used = false;
        // if (indices_left.size() == 0 || indices_right.size() == 0) {
        //     // Fallback to equal partitions
        //     auto&& [f_split_value, f_indices_left, f_indices_right]
        //       = equal_partitions(frame, axis_idx, positions, avg_min_ext);
        //     split_value = f_split_value;
        //     indices_left = std::move(f_indices_left);
        //     indices_right = std::move(f_indices_right);
        //     fallback_used = true;
        // }

        stats.add_split(fallback_used);

        auto [left_box, right_box] = frame.box.split(axis, split_value);
        // for (auto ix : indices_left) {
        //     if (left_box.max[axis_idx] < positions[ix][axis_idx]) {
        //         left_box.max[axis_idx] = positions[ix][axis_idx];
        //     }
        // }
        // for (auto ix : indices_right) {
        //     if (right_box.min[axis_idx] > positions[ix][axis_idx]) {
        //         right_box.min[axis_idx] = positions[ix][axis_idx];
        //     }
        // }

        stack.emplace(StackFrame{std::move(indices_left), left_box, axis});
        stack.emplace(StackFrame{std::move(indices_right), right_box, axis});
    }

    subsampled_pointcloud.resize(num_output_points);
    debug_out.resize(py::ssize_t(debug_out_idx));

    stats.stop_timer();
    stats.print();

    return std::make_pair(subsampled_pointcloud, debug_out);
}
} // namespace mdi::pointcloud
