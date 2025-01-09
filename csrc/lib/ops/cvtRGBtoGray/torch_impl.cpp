#include <ATen/TensorUtils.h>
#include <ATen/ops/zero.h>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <torch/torch.h>

#include "../torch_impl.hpp"
#include "pmpp/types/cxx_types.hpp"

namespace pmpp::ops::cpu
{
extern void launchCvtRGBtoGray(uint8_t* picOut, const uint8_t* picIn,
                               uint32_t nRows, uint32_t nCols);
namespace torch_impl
{
auto cvtRGBtoGrayImpl(const torch::Tensor& img) -> torch::Tensor
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

namespace pmpp::ops::cuda
{
extern void launchCvtRGBtoGray(uint8_t* picOut, const uint8_t* picIn,
                               uint32_t nRows, uint32_t nCols);
namespace torch_impl
{
auto cvtRGBtoGrayImpl(const torch::Tensor& img) -> torch::Tensor
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