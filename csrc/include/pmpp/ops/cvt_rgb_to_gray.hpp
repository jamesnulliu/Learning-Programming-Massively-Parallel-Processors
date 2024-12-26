#pragma once

#include "pmpp/system.hpp"
#include "pmpp/types/cxx_types.hpp"
#include "pmpp/types/torch_types.hpp"

namespace pmpp::ops
{

template <DeviceType DeviceT>
void launchCvtRGBtoGray(uint8_t* picOut, const uint8_t* picIn, uint32_t width,
                        uint32_t height) = delete;

template <>
PMPP_API void launchCvtRGBtoGray<DeviceType::CUDA>(uint8_t* picOut,
                                                   const uint8_t* picIn,
                                                   uint32_t width,
                                                   uint32_t height);

template <>
PMPP_API void launchCvtRGBtoGray<DeviceType::CPU>(uint8_t* picOut,
                                                  const uint8_t* picIn,
                                                  uint32_t width,
                                                  uint32_t height);
}  // namespace pmpp::ops
