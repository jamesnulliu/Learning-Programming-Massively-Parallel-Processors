#pragma once

#include <type_traits>

namespace pmpp
{
/**
 * @brief Calculate the ceiling of the division of two integers.
 *
 * @tparam T1 The type of the dividend.
 * @tparam T2 The type of the divisor.
 * @param a The dividend.
 * @param b The divisor.
 * @return The ceiling of the division of `a` by `b`.
 */
template <typename T1, typename T2>
    requires std::is_integral_v<T1> && std::is_integral_v<T2>
[[nodiscard]] constexpr auto ceilDiv(T1 a, T2 b) -> T1
{
    return T1((a + b - 1) / b);
}
}  // namespace pmpp