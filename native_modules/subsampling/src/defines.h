#ifndef NATIVE_MODULES_SUBSAMPLING_SRC_DEFINES
#define NATIVE_MODULES_SUBSAMPLING_SRC_DEFINES

#include <cstdint>
#include <Eigen/Core>

namespace mdi {
using FloatT = float;
using IntT = int32_t;

using FVector3D = Eigen::Matrix<FloatT, 3, 1>;
using FVector4D = Eigen::Matrix<FloatT, 4, 1>;
using FMatrix3D = Eigen::Matrix<FloatT, 3, 3>;
using FMatrix3x4 = Eigen::Matrix<FloatT, 3, 4>;
using FMatrix4D = Eigen::Matrix<FloatT, 4, 4>;
} // namespace mdi

#endif /* NATIVE_MODULES_SUBSAMPLING_SRC_DEFINES */
