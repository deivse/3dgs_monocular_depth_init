#include "raster_utils.h"

#include <algorithm>

namespace mdi::raster {
namespace {
    bool horizontal_line_fast(int x1, int x2, const int y, const std::function<bool(IntT, IntT)>& func) {
        for (; x1 <= x2; ++x1)
            if (!func(x1, y)) return false;
        return true;
    }

    bool vertical_line_fast(int y1, int y2, const int x, const std::function<bool(IntT, IntT)>& func) {
        for (; y1 <= y2; ++y1)
            if (!func(x, y1)) return false;
        return true;
    }

    bool main_axis_x(const IVector2D start, const IVector2D end, const int stepY, const int absDeltaY,
                     const int absDeltaX, const std::function<bool(IntT, IntT)>& func) {
        const auto k1 = 2 * absDeltaY;
        const auto k2 = 2 * (absDeltaY - absDeltaX);

        int p = 2 * absDeltaY - absDeltaX;

        auto currentCoords = start;
        if (!func(currentCoords.x(), currentCoords.y())) return false;

        while (currentCoords[0] < end[0]) {
            currentCoords[0]++;
            if (p > 0) {
                currentCoords[1] += stepY;
                p += k2;
            } else {
                p += k1;
            }
            if (!func(currentCoords.x(), currentCoords.y())) return false;
        }
        return true;
    }

    bool main_axis_y(const IVector2D start, const IVector2D end, const int stepX, const int absDeltaY,
                     const int absDeltaX, const std::function<bool(IntT, IntT)>& func) {
        const auto k1 = 2 * absDeltaX;
        const auto k2 = 2 * (absDeltaX - absDeltaY);

        int p = 2 * absDeltaX - absDeltaY;

        auto currentCoords = start;
        if (!func(currentCoords.x(), currentCoords.y())) return false;

        while (currentCoords[1] < end[1]) {
            currentCoords[1]++;
            if (p > 0) {
                currentCoords[0] += stepX;
                p += k2;
            } else {
                p += k1;
            }
            if (!func(currentCoords.x(), currentCoords.y())) return false;
        }
        return true;
    }

    struct Edge
    {
        IVector2D m_Bottom;
        int m_TopY;
        float m_InverseSlope;

        Edge() = default;
        Edge(IVector2D bottom, IVector2D top) : m_Bottom(bottom), m_TopY(top.y()) {
            m_InverseSlope = (top.x() - bottom.x()) / static_cast<float>(top.y() - bottom.y());
        }

        int findIntersectionWithScanLineX(int y) const { return m_InverseSlope * (y - m_Bottom.y()) + m_Bottom.x(); }
    } __attribute__((aligned(16)));
} // namespace

bool for_each_triangle_pixel(FVector2D v0, FVector2D v1, FVector2D v2,
                             const std::function<bool(IntT x, IntT y)>& func) {
    auto v0_i = IVector2D(static_cast<IntT>(std::round(v0.x())), static_cast<IntT>(std::round(v0.y())));
    auto v1_i = IVector2D(static_cast<IntT>(std::round(v1.x())), static_cast<IntT>(std::round(v1.y())));
    auto v2_i = IVector2D(static_cast<IntT>(std::round(v2.x())), static_cast<IntT>(std::round(v2.y())));

    std::array<IVector2D, 3> sorted{v0_i, v1_i, v2_i};
    std::sort(sorted.begin(), sorted.end(), [](IVector2D a, IVector2D b) { return a.y() < b.y(); });

    Edge longEdge{sorted[0], sorted[2]};
    Edge topEdge{sorted[1], sorted[2]};
    Edge bottomEdge{sorted[0], sorted[1]};

    int startY = bottomEdge.m_Bottom.y();
    int endY = bottomEdge.m_TopY;

    for (int y = startY; y < endY; y++) {
        int start = longEdge.findIntersectionWithScanLineX(y);
        int end = bottomEdge.findIntersectionWithScanLineX(y);
        if (start > end) std::swap(start, end);

        for (int x = start; x <= end; ++x) {
            if (!func(x, y)) return false;
        }
    }

    startY = topEdge.m_Bottom.y();
    endY = topEdge.m_TopY;

    for (int y = startY; y < endY; y++) {
        int start = longEdge.findIntersectionWithScanLineX(y);
        int end = topEdge.findIntersectionWithScanLineX(y);
        if (start > end) std::swap(start, end);

        for (int x = start; x <= end; ++x) {
            if (!func(x, y)) return false;
        }
    }
    return true;
}

bool for_each_line_pixel(IVector2D start, IVector2D end, const std::function<bool(IntT, IntT)>& func) {
    if (start == end) {
        return func(start.x(), start.y());
    }

    const auto deltaX = end.x() - start.x();
    const auto absDeltaX = std::abs(deltaX);
    const auto deltaY = end.y() - start.y();
    const auto absDeltaY = std::abs(deltaY);

    /// @returns -1 if secondaryAxisDelta < 0, 1 otherwise
    constexpr auto calcSecondaryAxisStep = [](int secondaryAxisDelta) { return -2 * (secondaryAxisDelta < 0) + 1; };

    if (absDeltaY < absDeltaX) {
        if (deltaX > 0)
            return main_axis_x(start, end, calcSecondaryAxisStep(deltaY), absDeltaY, absDeltaX, func);
        else if (deltaX < 0)
            // deltaY := -deltaY since points swapped
            return main_axis_x(end, start, calcSecondaryAxisStep(-deltaY), absDeltaY, absDeltaX, func);
        else {
            if (deltaY > 0)
                return vertical_line_fast(start.y(), end.y(), start.x(), func);
            else
                return vertical_line_fast(end.y(), start.y(), start.x(), func);
        }
    } else {
        if (deltaY > 0)
            return main_axis_y(start, end, calcSecondaryAxisStep(deltaX), absDeltaY, absDeltaX, func);
        else if (deltaY < 0)
            // deltaX := -deltaX since points swapped
            return main_axis_y(end, start, calcSecondaryAxisStep(-deltaX), absDeltaY, absDeltaX, func);
        else {
            if (deltaX > 0)
                return horizontal_line_fast(start.x(), end.x(), start.y(), func);
            else
                return horizontal_line_fast(end.x(), start.x(), start.y(), func);
        }
    }
}
} // namespace mdi::raster
