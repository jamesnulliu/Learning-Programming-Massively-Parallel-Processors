#include <algorithm>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "pmpp/types/cxx_types.hpp"
#include "pmpp/types/extented_concepts.hpp"

namespace pmpp::ops::cpu
{
template <typename T1, typename T2>
[[nodiscard]] auto vecCompare(T1 vec1, T2 vec2, size_t n = size_t(-1)) -> bool
{
    if constexpr (is_range<T1> && is_range<T2>) {
        return std::ranges::equal(vec1, vec2);
    } else if constexpr (std::is_pointer_v<T1> && std::is_pointer_v<T2>) {
        return std::ranges::equal(std::span(vec1, n), std::span(vec2, n));
    } else {
        throw std::runtime_error("Unsupported type for vec compare.");
    }
}
}  // namespace pmpp::ops::cpu