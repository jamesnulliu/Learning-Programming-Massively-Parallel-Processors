#pragma once

#include <type_traits>

namespace pmpp
{
template <typename T1, typename T2>
    requires std::is_integral_v<T1> && std::is_integral_v<T2>
constexpr auto ceilDiv(T1 a, T2 b) -> T1
{
    return T1((a + b - 1) / b);
}
}  // namespace pmpp