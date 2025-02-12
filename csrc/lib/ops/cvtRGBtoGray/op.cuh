#pragma once
#include "pmpp/pch.hpp"

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

namespace torch_impl
{
inline auto cvtRGBtoGray(const torch::Tensor& img) -> torch::Tensor
{
    TORCH_CHECK(img.scalar_type() == torch::kUInt8,
                "Expected in Tensor to have dtype = torch::kUInt8, but have: ",
                torch::toString(img.scalar_type()));

    auto nRows = img.size(0);
    auto nCols = img.size(1);

    torch::Tensor imgOut = torch::zeros(
        {nRows, nCols},
        torch::TensorOptions().dtype(torch::kUInt8).device(img.device()));

    pmpp::ops::cuda::launchCvtRGBtoGray(imgOut.data_ptr<uint8_t>(),
                                        img.data_ptr<uint8_t>(), nRows, nCols);
    return imgOut;
}
}  // namespace torch_impl
}  // namespace pmpp::ops::cuda
