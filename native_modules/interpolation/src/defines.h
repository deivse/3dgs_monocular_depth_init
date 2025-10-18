#ifndef NATIVE_MODULES_INTERP_SRC_DEFINES
#define NATIVE_MODULES_INTERP_SRC_DEFINES

#include <cstdint>
#include <Eigen/Core>

namespace mdi {
using FloatT = float;
using IntT = int32_t;

using FVector2D = Eigen::Matrix<FloatT, 2, 1>;
} // namespace mdi

#endif /* NATIVE_MODULES_INTERP_SRC_DEFINES */
