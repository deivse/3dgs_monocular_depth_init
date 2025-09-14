#ifndef NATIVE_MODULES_SUBSAMPLING_SRC_DEBUG
#define NATIVE_MODULES_SUBSAMPLING_SRC_DEBUG

#include <Eigen/Core>

#include "defines.h"

namespace mdi::debug {
static const auto vis_colors = std::array{
  FVector3D{1.0f, 0.0f, 0.0f},  // Red
  FVector3D{0.0f, 1.0f, 0.0f},  // Green
  FVector3D{0.0f, 0.0f, 1.0f},  // Blue
  FVector3D{1.0f, 1.0f, 0.0f},  // Yellow
  FVector3D{1.0f, 0.0f, 1.0f},  // Magenta
  FVector3D{0.0f, 1.0f, 1.0f},  // Cyan
  FVector3D{1.0f, 0.5f, 0.0f},  // Orange
  FVector3D{0.5f, 0.0f, 0.5f},  // Purple
  FVector3D{0.5f, 0.5f, 0.5f},  // Gray
  FVector3D{0.5f, 0.5f, 0.0f},  // Olive
  FVector3D{0.0f, 0.5f, 0.5f},  // Teal
  FVector3D{0.5f, 0.0f, 0.0f},  // Maroon
  FVector3D{0.0f, 0.5f, 0.0f},  // Dark Green
  FVector3D{0.0f, 0.0f, 0.5f},  // Navy
  FVector3D{1.0f, 0.75f, 0.8f}, // Pink
  FVector3D{0.8f, 0.4f, 0.0f},  // Brown
  FVector3D{0.7f, 0.7f, 0.2f},  // Light Olive
  FVector3D{0.2f, 0.7f, 0.7f},  // Light Teal
  FVector3D{0.7f, 0.2f, 0.7f}   // Violet
};

inline FVector3D get_debug_color(size_t idx) { return vis_colors[idx % vis_colors.size()]; }

inline FVector3D get_debug_color_muted(size_t idx) {
    return vis_colors[idx % vis_colors.size()] * 0.25f + FVector3D{0.4f, 0.4f, 0.4f} * 0.75f;
}

} // namespace mdi::debug

#endif /* NATIVE_MODULES_SUBSAMPLING_SRC_DEBUG */
