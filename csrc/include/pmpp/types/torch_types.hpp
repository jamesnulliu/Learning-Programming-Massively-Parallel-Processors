#pragma once

#include <torch/torch.h>

#include "pmpp/types/cu_types.cuh"
#include "pmpp/types/cxx_types.hpp"

namespace pmpp
{
using DeviceType = torch::DeviceType;
static_assert(sizeof(DeviceType) == 1);
using ScalarType = torch::ScalarType;
static_assert(sizeof(ScalarType) == 1);

/**
 * @brief Combine DeviceT and ScalarT to a uint16_t.
 */
constexpr auto combineDeviceTandScalarT(DeviceType deviceT,
                                        ScalarType scalarT) -> uint16_t
{
    return (uint16_t(deviceT) << 8) | uint16_t(scalarT);
}
}  // namespace pmpp