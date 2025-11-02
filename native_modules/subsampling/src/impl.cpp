#include "impl.h"
#include "debug.h"
#include "geometry.h"
#include "stack.h"
#include "stat_accumulator.h"
#include <common/util.h>

#include <Eigen/Dense>
#include <iostream>
#include <ranges>
#include <algorithm>
#include <span>

namespace mdi::pointcloud {

namespace {
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

    struct ComputeMinGaussianExtentsStats
    {
        std::chrono::high_resolution_clock::time_point start_time, end_time;
        std::chrono::duration<double> duration;
        size_t num_points = 0;
        size_t num_unobserved_points = 0;

        void start_timer() { start_time = std::chrono::high_resolution_clock::now(); }
        void stop_timer() {
            end_time = std::chrono::high_resolution_clock::now();
            duration = end_time - start_time;
        }

        void add_point(bool observed) {
            num_points++;
            if (!observed) {
                num_unobserved_points++;
            }
        }

        std::string to_string() const {
            std::ostringstream oss;
            oss << "ComputeMinGaussianExtentsStats:\n";
            oss << "  Time taken: " << duration.count() << " seconds\n";
            oss << "  Number of points: " << num_points << "\n";
            oss << "  Number of unobserved points: " << num_unobserved_points << "\n";
            return oss.str();
        }
        void print() const { std::cout << to_string(); }
        void save() const { common::dump_str_to_file("min_gaussian_extents_stats.txt", to_string()); }
    };
} // namespace

std::vector<FloatT> compute_minimal_gaussian_extents(const py::array_t<FloatT>& points,
                                                     const std::vector<FMatrix3D>& intrinsic_matrices,
                                                     const std::vector<FMatrix3x4>& proj_matrices,
                                                     const py::array_t<IntT>& image_sizes) {
    ComputeMinGaussianExtentsStats stats{};
    stats.start_timer();

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
        if (extent != std::numeric_limits<FloatT>::max()) {
            stats.add_point(true);
        } else {
            extent = static_cast<FloatT>(-1.0); // No valid observation, set extent to -1
            stats.add_point(false);
        }
    }

    stats.stop_timer();
    stats.print();
    stats.save();

    return min_gaussian_extents;
}

namespace {
    struct SubsamplePointcloudStats
    {
        StatAccumulator bbox_aspect_ratio;
        StatAccumulator bbox_min_size;
        StatAccumulator bbox_max_size;
        StatAccumulator merged_min_gaussian_extent;
        StatAccumulator merged_num_points_in_leaf;
        std::chrono::high_resolution_clock::time_point start_time, end_time;
        std::chrono::duration<double> duration;
        size_t total_splits = 0;
        size_t total_fallback_splits = 0;
        size_t num_merged_points = 0;
        size_t num_unmerged_points = 0;
        size_t initial_num_points = 0;
        size_t final_num_points = 0;

        void add_merged(FloatT aspect, const FVector3D& tight_box_diag, FloatT min_gaussian_extent,
                        size_t num_points_in_leaf) {
            bbox_aspect_ratio.add(aspect);
            bbox_min_size.add(tight_box_diag.minCoeff());
            bbox_max_size.add(tight_box_diag.maxCoeff());
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

        std::string to_string() const {
            std::ostringstream oss;
            oss << "SubsamplePointcloudStats:\n";
            oss << "  Subsampled from " << initial_num_points << " to " << final_num_points << " points.\n";
            oss << "  Merged points: " << num_merged_points << "\n";
            oss << "  Unmerged points: " << num_unmerged_points << "\n";
            oss << "  Time taken: " << duration.count() << " seconds\n";
            if (num_merged_points > 0) {
                oss << "  BBox aspect ratio: mean=" << bbox_aspect_ratio.mean()
                    << ", stddev=" << bbox_aspect_ratio.stddev() << ", min=" << bbox_aspect_ratio.min
                    << ", max=" << bbox_aspect_ratio.max << "\n";
                oss << "  BBox min size: mean=" << bbox_min_size.mean() << ", stddev=" << bbox_min_size.stddev()
                    << ", min=" << bbox_min_size.min << ", max=" << bbox_min_size.max << "\n";
                oss << "  BBox max size: mean=" << bbox_max_size.mean() << ", stddev=" << bbox_max_size.stddev()
                    << ", min=" << bbox_max_size.min << ", max=" << bbox_max_size.max << "\n";
                oss << "  Merged min gaussian extent: mean=" << merged_min_gaussian_extent.mean()
                    << ", stddev=" << merged_min_gaussian_extent.stddev() << ", min=" << merged_min_gaussian_extent.min
                    << ", max=" << merged_min_gaussian_extent.max << "\n";
                oss << "  Merged num points in leaf: mean=" << merged_num_points_in_leaf.mean()
                    << ", stddev=" << merged_num_points_in_leaf.stddev() << ", min=" << merged_num_points_in_leaf.min
                    << ", max=" << merged_num_points_in_leaf.max << "\n";
                oss << "  Total splits: " << total_splits
                    << ", of which fallback strat used in: " << total_fallback_splits << "\n";
            }
            return oss.str();
        }
        void save() const { common::dump_str_to_file("subsample_pointcloud_stats.txt", to_string()); }
        void print() const { std::cout << to_string(); }
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
        bool fallback_used = false;

        SplitResult mark_fallback() {
            fallback_used = true;
            return *this;
        }
    };

    // Meh
    SplitResult equal_num_pts(StackFrame& frame, size_t axis_idx, const std::vector<FVector3D>& positions,
                              FloatT /*avg_min_extent*/) {
        auto mid_it = frame.indices.begin() + frame.indices.size() / 2;
        std::ranges::nth_element(frame.indices, mid_it,
                                 [&](IndexT a, IndexT b) { return positions[a][axis_idx] < positions[b][axis_idx]; });
        FloatT split_value = (positions[*(mid_it - 1)][axis_idx] + positions[*mid_it][axis_idx]) / FloatT(2);

        std::vector<IndexT> indices_left(frame.indices.begin(), mid_it);
        std::vector<IndexT> indices_right(mid_it, frame.indices.end());
        return {split_value, std::move(indices_left), std::move(indices_right)};
    }

    // Works best in practice
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

    // Doesn't really work
    SplitResult max_gap(StackFrame& frame, size_t axis_idx, const std::vector<FVector3D>& positions,
                        FloatT avg_min_extent) {
        FloatT dist_max = std::numeric_limits<FloatT>::lowest();
        FloatT mean_dist = 0;
        size_t split_idx = 0;
        std::vector<uint32_t> indices_left, indices_right;
        std::ranges::sort(frame.indices, [&positions, axis_idx](uint32_t a, uint32_t b) {
            return positions[a][axis_idx] < positions[b][axis_idx];
        });

        for (size_t i = 1; i < frame.indices.size(); ++i) {
            FloatT dist_to_prev
              = std::abs(positions[frame.indices[i - 1]][axis_idx] - positions[frame.indices[i]][axis_idx]);
            mean_dist += dist_to_prev;
            if (dist_to_prev > dist_max) {
                dist_max = dist_to_prev;
                split_idx = i;
            }
        }
        mean_dist /= static_cast<FloatT>(frame.indices.size() - 1);

        if (dist_max < 1.5 * mean_dist) {
            // Bad split, use spatial median instead
            return spatial_median(frame, axis_idx, positions, avg_min_extent).mark_fallback();
        }
        FloatT split_value
          = (positions[frame.indices[split_idx - 1]][axis_idx] + positions[frame.indices[split_idx]][axis_idx])
            / FloatT(2);

        indices_left = std::vector<uint32_t>(frame.indices.begin(), frame.indices.begin() + split_idx);
        indices_right = std::vector<uint32_t>(frame.indices.begin() + split_idx, frame.indices.end());

        return {split_value, std::move(indices_left), std::move(indices_right)};
    }
} // namespace

std::pair<PointCloud, PointCloud> subsample_pointcloud_impl(const PointCloud& pointcloud,
                                                            const std::vector<FloatT>& min_gaussian_extents,
                                                            FloatT max_bbox_aspect_ratio, FloatT min_extent_mult) {
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
    stack.emplace(
      StackFrame{make_index_sequence(positions.size()), geom::BoundingBox::cube_from_points(positions), geom::Axis::X});

    while (!stack.empty()) {
        auto frame = stack.pop();
        if (frame.indices.size() == 0) {
            continue;
        }
        if (frame.indices.size() == 1) {
            auto ix = frame.indices[0];
            stats.add_unmerged();
            output_point(positions[ix], *reinterpret_cast<const FVector3D*>(unchecked_rgbs.data(ix, 0)));
            continue;
        }

        FloatT avg_min_ext = compute_average_min_gaussian_extent(frame.indices, min_gaussian_extents);

        const auto tight_bbox = geom::BoundingBox::from_points(positions, frame.indices);

        const auto original_box_aspect_ratio = frame.box.diagonal().maxCoeff() / frame.box.diagonal().minCoeff();
        const auto tight_box_aspect_ratio = tight_bbox.diagonal().maxCoeff() / tight_bbox.diagonal().minCoeff();

        const FloatT aspect_ratio = std::min(original_box_aspect_ratio, tight_box_aspect_ratio);
        const auto conservative_threshold = tight_bbox.diagonal().maxCoeff();

        if (aspect_ratio <= max_bbox_aspect_ratio && conservative_threshold <= min_extent_mult * avg_min_ext) {
            // Merge points
            FVector3D merged_pos = FVector3D::Zero();
            FVector3D merged_rgb = FVector3D::Zero();
            for (auto ix : frame.indices) {
                merged_pos += positions[ix];
                merged_rgb += *reinterpret_cast<const FVector3D*>(unchecked_rgbs.data(ix, 0));

                output_debug(positions[ix], mdi::debug::get_debug_color_muted(debug_group_idx));
            }
            merged_pos /= static_cast<FloatT>(frame.indices.size());
            merged_rgb /= static_cast<FloatT>(frame.indices.size());

            stats.add_merged(aspect_ratio, tight_bbox.diagonal(), min_extent_mult * avg_min_ext, frame.indices.size());
            output_point(merged_pos, merged_rgb);
            output_debug(merged_pos, mdi::debug::get_debug_color(debug_group_idx++));
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

        auto [split_value, indices_left, indices_right, fallback_used]
          = split_fn(frame, axis_idx, positions, avg_min_ext);

        stats.add_split(fallback_used);

        auto [left_box, right_box] = frame.box.split(axis, split_value);

        stack.emplace(StackFrame{std::move(indices_left), left_box, axis});
        stack.emplace(StackFrame{std::move(indices_right), right_box, axis});
    }

    subsampled_pointcloud.resize(num_output_points);
    debug_out.resize(py::ssize_t(debug_out_idx));

    stats.initial_num_points = pointcloud.positions.shape(0);
    stats.final_num_points = subsampled_pointcloud.positions.shape(0);
    stats.stop_timer();
    stats.print();
    stats.save();

    return std::make_pair(subsampled_pointcloud, debug_out);
}
} // namespace mdi::pointcloud
