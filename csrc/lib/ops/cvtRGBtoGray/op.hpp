#pragma once
#include "pmpp/pch.hpp"

namespace pmpp::ops::cpu
{
void launchCvtRGBtoGray(uint8_t* picOut, const uint8_t* picIn, uint32_t nRows,
                        uint32_t nCols)
{
#pragma omp for
    for (uint32_t i = 0; i < nRows * nCols; ++i) {
        uint8_t r = picIn[i * 3];
        uint8_t g = picIn[i * 3 + 1];
        uint8_t b = picIn[i * 3 + 2];
        picOut[i] = uint8_t(0.21F * r + 0.71F * g + 0.07F * b);
    }
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

    pmpp::ops::cpu::launchCvtRGBtoGray(imgOut.mutable_data_ptr<uint8_t>(),
                                       img.const_data_ptr<uint8_t>(), nRows,
                                       nCols);

    return imgOut;
}
}  // namespace torch_impl
}  // namespace pmpp::ops::cpu
