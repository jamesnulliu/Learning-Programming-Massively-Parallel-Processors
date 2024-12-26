#include <torch/torch.h>

#include "pmpp/ops/cvt_rgb_to_gray.hpp"

namespace pmpp::ops::torch_impl
{

auto cvtRGBtoGrayCpuImpl(const torch::Tensor& img) -> torch::Tensor
{
    TORCH_CHECK(img.scalar_type() == torch::kUInt8,
                "Expected in Tensor to have dtype = torch::kUInt8, but have: ",
                torch::toString(img.scalar_type()));

    auto width = img.size(0);
    auto height = img.size(1);

    torch::Tensor imgOut = torch::empty_like(img);

    pmpp::ops::launchCvtRGBtoGray<DeviceType::CPU>(
        imgOut.mutable_data_ptr<uint8_t>(), img.const_data_ptr<uint8_t>(),
        width, height);

    return imgOut;
}

auto cvtRGBtoGrayCudaImpl(const torch::Tensor& img) -> torch::Tensor
{
    TORCH_CHECK(img.scalar_type() == torch::kUInt8,
                "Expected in Tensor to have dtype = torch::kUInt8, but have: ",
                torch::toString(img.scalar_type()));

    auto width = img.size(0);
    auto height = img.size(1);

    torch::Tensor imgOut = torch::empty_like(img);

    pmpp::ops::launchCvtRGBtoGray<DeviceType::CUDA>(
        imgOut.mutable_data_ptr<uint8_t>(), img.const_data_ptr<uint8_t>(),
        width, height);

    return imgOut;
}
}  // namespace pmpp::ops::torch_impl
