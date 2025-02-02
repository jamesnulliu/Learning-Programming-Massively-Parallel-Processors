#include <cuda_runtime.h>

#include "pmpp/types/cxx_types.hpp"
#include "pmpp/utils/address.hpp"
#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{

__global__ void cvtRGBtoGrayKernel(uint8_t* outImg, const uint8_t* inImg,
                                   uint32_t height, uint32_t width)
{
    // Suppose each pixel is 3 consecutive chars for the 3 channels (RGB).
    constexpr uint32_t N_CHANNELS = 3;
    // Assign each cuda thread to process one pixel.
    uint32_t row = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row >= height || col >= width) {
        return;
    }

    auto grayOffset = offset<uint32_t>(row, col, height, width);
    uint32_t rgbOffset = grayOffset * N_CHANNELS;

    uint8_t r = inImg[rgbOffset];
    uint8_t g = inImg[rgbOffset + 1];
    uint8_t b = inImg[rgbOffset + 2];

    outImg[grayOffset] = uint8_t(0.21F * r + 0.71F * g + 0.07F * b);
}

void launchCvtRGBtoGray(uint8_t* outImg, const uint8_t* inImg, uint32_t nRows,
                        uint32_t nCols)
{
    dim3 blockSize = {32, 32};
    dim3 gridSize = {ceilDiv(nRows, 32), ceilDiv(nCols, 32)};

    cvtRGBtoGrayKernel<<<gridSize, blockSize>>>(outImg, inImg, nRows, nCols);

    PMPP_DEBUG_CUDA_ERR_CHECK(cudaGetLastError());
}

}  // namespace pmpp::ops::cuda
