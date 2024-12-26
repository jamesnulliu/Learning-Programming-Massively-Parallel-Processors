#include <cuda_runtime.h>

#include "pmpp/ops/cvt_rgb_to_gray.hpp"
#include "pmpp/utils/address.hpp"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops
{

__global__ void cvtRGBtoGrayKernel(uint8_t* picOut, const uint8_t* picIn,
                                   uint32_t width, uint32_t height)
{
    // Suppose each pixel is 3 consecutive chars for the 3 channels (RGB).
    constexpr uint32_t N_CHANNELS = 3;
    // Assign each cuda thread to process one pixel.
    uint32_t col = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col >= width || row >= height) {
        return;
    }

    auto grayOffset = computeOffset<uint32_t>(col, width, width, height);
    uint32_t rgbOffset = grayOffset * N_CHANNELS;

    uint8_t r = picIn[rgbOffset];
    uint8_t g = picIn[rgbOffset + 1];
    uint8_t b = picIn[rgbOffset + 2];

    picOut[grayOffset] = uint8_t(0.21F * r + 0.71F * g + 0.07F * b);
}

template <>
void launchCvtRGBtoGray<DeviceType::CUDA>(uint8_t* picOut, const uint8_t* picIn,
                                          uint32_t width, uint32_t height)
{
    dim3 blockSize = {256, 256};
    dim3 gridSize = {ceil(width, 256), ceil(height, 256)};

    cvtRGBtoGrayKernel<<<gridSize, blockSize>>>(picOut, picIn, width, height);
}

}  // namespace pmpp::ops