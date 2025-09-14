#ifndef NATIVE_MODULES_SUBSAMPLING_SRC_GEOMETRY
#define NATIVE_MODULES_SUBSAMPLING_SRC_GEOMETRY

#include "defines.h"

namespace mdi::geom {

template<typename R, typename T>
concept forward_range_of = std::ranges::forward_range<R> && std::same_as<std::ranges::range_value_t<R>, T>;

enum class Axis : uint8_t
{
    X,
    Y,
    Z,
    NUM_AXES
};

inline constexpr uint8_t axis_to_int(Axis axis) { return static_cast<uint8_t>(axis); }
inline constexpr Axis axis_from_int(uint8_t axis) { return static_cast<Axis>(axis); }

constexpr auto NumAxes = axis_to_int(Axis::NUM_AXES);

struct BoundingBox
{
    FVector3D min, max;

    static BoundingBox from_points(forward_range_of<FVector3D> auto&& points) {
        const auto min_val = std::numeric_limits<FVector3D::value_type>::lowest();
        const auto max_val = std::numeric_limits<FVector3D::value_type>::max();
        BoundingBox retval{.min = {max_val, max_val, max_val}, .max = {min_val, min_val, min_val}};

        for (auto&& point : points) {
            retval.min = retval.min.cwiseMin(point);
            retval.max = retval.max.cwiseMax(point);
        }
        return retval;
    }

    FVector3D diagonal() const { return max - min; }

    std::pair<BoundingBox, BoundingBox> split(Axis axis, FloatT split_value) const {
        const auto axis_idx = axis_to_int(axis);
        BoundingBox box1 = *this;
        box1.max[axis_idx] = split_value;
        BoundingBox box2 = *this;
        box2.min[axis_idx] = split_value;
        return {box1, box2};
    }

    Axis longest_axis() const {
        const auto diag = diagonal();
        if (diag.x() >= diag.y() && diag.x() >= diag.z()) {
            return Axis::X;
        } else if (diag.y() >= diag.x() && diag.y() >= diag.z()) {
            return Axis::Y;
        } else {
            return Axis::Z;
        }
    }
};

} // namespace mdi::geom

#endif /* NATIVE_MODULES_SUBSAMPLING_SRC_GEOMETRY */
