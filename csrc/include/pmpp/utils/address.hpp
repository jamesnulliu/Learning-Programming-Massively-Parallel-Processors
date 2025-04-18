#pragma once

#include <cassert>
#include <cstddef>

#if __cplusplus < 202002L
    #include <array>
#else
    #include <tuple>
    #include <utility>
#endif

namespace pmpp
{
/**
 * @brief Compute the offset of a multi-dimensional array.
 *   A typical use case is that if you have rowIdx, colIdx, nRows and nCols,
 *   to calculate the linear index of the element at (rowIdx, colIdx), you can
 *   use this function as follows:
 *   > offset(rowIdx, colIdx, nRows, nCols)
 *
 * @param args First half is the indices, second half is the size of each
 *             dimension.
 * @return std::uint32_t The offset of the multi-dimensional array.
 *
 * @example
 *   1. To calculate the offset of idx (2, 1) in a 2D array of dim (4, 3):
 *      > offset(2, 1, 4, 3) -> 1*1 + 2*3 = 7
 *   2. To calculate the offset of idx (1, 2, 3) in a 3D array of dim
 *      (4, 5, 6):
 *      > offset(1, 2, 3, 4, 5, 6) -> 3*1 + 2*6 + 1*6*5 = 45
 */
template <typename OffsetT, typename... ArgsT>
[[nodiscard]] constexpr auto offset(ArgsT... args) -> OffsetT
{
    constexpr std::size_t nArgs = sizeof...(ArgsT);
    constexpr std::size_t nDims = nArgs / 2;

    OffsetT offset = 0, stride = 1;

#if __cplusplus >= 202002L
    auto params = std::make_tuple(static_cast<OffsetT>(args)...);
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((I < nDims ? (offset += std::get<nDims - 1 - I>(params) * stride,
                       stride *= std::get<nArgs - 1 - I>(params))
                    : 0),
         ...);
    }(std::make_index_sequence<nDims>{});
#else
    auto params = std::array{static_cast<OffsetT>(args)...};
    for (std::size_t i = 0; i < nDims; ++i) {
        offset += params[nDims - 1 - i] * stride;
        stride *= params[nArgs - 1 - i];
    }
#endif

    return offset;
}

}  // namespace pmpp