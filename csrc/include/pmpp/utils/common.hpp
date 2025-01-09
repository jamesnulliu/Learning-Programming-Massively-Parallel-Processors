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

}  // namespace pmpp