#include <cstdio>
#include <cuda_runtime_api.h>
#include <torch/torch.h>

#include "../ops.hpp"
#include "../torch_impl.hpp"

namespace pmpp::ops::cpu::torch_impl
{
auto cvtRGBtoGray(const torch::Tensor& img) -> torch::Tensor
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
}  // namespace pmpp::ops::cpu::torch_impl

namespace pmpp::ops::cuda::torch_impl
{
auto cvtRGBtoGray(const torch::Tensor& img) -> torch::Tensor
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
}  // namespace pmpp::ops::cuda::torch_impl
