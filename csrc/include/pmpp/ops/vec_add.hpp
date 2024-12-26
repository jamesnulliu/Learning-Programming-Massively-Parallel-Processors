#pragma once

#include "pmpp/system.hpp"
#include "pmpp/types/cxx_types.hpp"
#include "pmpp/types/torch_types.hpp"

namespace pmpp::ops
{

template <DeviceType DeviceT, typename ScalarT>
void launchVecAdd(const ScalarT* a, const ScalarT* b, ScalarT* c,
                  size_t n) = delete;

template <>
PMPP_API void launchVecAdd<DeviceType::CPU>(const fp32_t* a, const fp32_t* b,
                                            fp32_t* c, size_t n);
template <>
PMPP_API void launchVecAdd<DeviceType::CUDA>(const fp32_t* a, const fp32_t* b,
                                             fp32_t* c, size_t n);

}  // namespace pmpp::ops