#include "pmpp/ops/cvt_rgb_to_gray.hpp"

namespace pmpp::ops
{
template <>
void launchCvtRGBtoGray<DeviceType::CPU>(uint8_t* picOut, const uint8_t* picIn,
                                         uint32_t width, uint32_t height)
{
#pragma omp for
    for (uint32_t i = 0; i < width * height; ++i) {
        uint8_t r = picIn[i * 3];
        uint8_t g = picIn[i * 3 + 1];
        uint8_t b = picIn[i * 3 + 2];
        picOut[i] = uint8_t(0.21F * r + 0.71F * g + 0.07F * b);
    }
}

}  // namespace pmpp::ops