#ifndef NATIVE_MODULES_SUBSAMPLING_SRC_STAT_ACCUMULATOR
#define NATIVE_MODULES_SUBSAMPLING_SRC_STAT_ACCUMULATOR

#include "defines.h"
namespace mdi {
struct StatAccumulator
{
    size_t count = 0;
    FloatT sum = 0;
    FloatT sum_of_squares = 0;
    FloatT min = std::numeric_limits<FloatT>::max();
    FloatT max = std::numeric_limits<FloatT>::lowest();

    void add(FloatT value) {
        count++;
        sum += value;
        sum_of_squares += value * value;
        min = std::min(min, value);
        max = std::max(max, value);
    }

    FloatT mean() const { return count > 0 ? sum / static_cast<FloatT>(count) : 0; }

    FloatT variance() const {
        if (count == 0) return 0;
        FloatT mean_value = mean();
        return (sum_of_squares / static_cast<FloatT>(count)) - (mean_value * mean_value);
    }

    FloatT stddev() const { return std::sqrt(variance()); }
};
} // namespace mdi

#endif /* NATIVE_MODULES_SUBSAMPLING_SRC_STAT_ACCUMULATOR */
