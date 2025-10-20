#ifndef NATIVE_MODULES_INTERPOLATION_SRC_RASTER_UTILS
#define NATIVE_MODULES_INTERPOLATION_SRC_RASTER_UTILS

#include "defines.h"

namespace mdi::raster {

bool for_each_triangle_pixel(FVector2D v0, FVector2D v1, FVector2D v2, const std::function<bool(IntT x, IntT y)>& func);
bool for_each_line_pixel(IVector2D p0, IVector2D p1, const std::function<bool(IntT x, IntT y)>& func);

} // namespace mdi::raster

#endif /* NATIVE_MODULES_INTERPOLATION_SRC_RASTER_UTILS */
