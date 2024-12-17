#pragma once
#include "pmpp/types/cxx_types.hpp"

namespace pmpp
{
enum class Device : int8_t
{
    CPU = 0,
    CUDA = 1,
};
}  // namespace pmpp