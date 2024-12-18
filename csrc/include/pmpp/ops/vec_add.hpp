#pragma once

#include "pmpp/system.hpp"
#include "pmpp/types/cxx_types.hpp"
#include "pmpp/types/device.hpp"

namespace pmpp::ops
{

using Device = ::pmpp::Device;

template <typename ScalarT, Device DeviceT>
void launchVecAdd(const ScalarT* a, const ScalarT* b, ScalarT* c,
                  size_t n) = delete;

template <>
PMPP_API void launchVecAdd<fp32_t, Device::CPU>(const fp32_t* a,
                                                const fp32_t* b, fp32_t* c,
                                                size_t n);
template <>
PMPP_API void launchVecAdd<fp32_t, Device::CUDA>(const fp32_t* a,
                                                 const fp32_t* b, fp32_t* c,
                                                 size_t n);

}  // namespace pmpp::ops