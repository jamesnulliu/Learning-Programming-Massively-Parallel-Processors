#include <cstddef>
#include <string>

#include "pmpp/types/concepts.hpp"

namespace pmpp
{

template <is_range ArrT>
auto arr2str(const ArrT& arr) -> std::string
{
    size_t i = 0;
    std::string str = "[";
    for (const auto& elem : arr) {
        if (i++ > 0) {
            str += ", ";
        }
        str += std::to_string(elem);
    }
    str += "]";
    return str;
}

template <typename T>
void initMemory(T* ptr, size_t n, const T& val)
{
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
}

}  // namespace pmpp