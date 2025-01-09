#pragma once

#include <torch/torch.h>

namespace pmpp::ops::cpu::torch_impl
{
auto vectorAddImpl(const torch::Tensor& A,
                   const torch::Tensor& B) -> torch::Tensor;
auto cvtRGBtoGrayImpl(const torch::Tensor& img) -> torch::Tensor;
}  // namespace pmpp::ops::cpu::torch_impl

namespace pmpp::ops::cuda::torch_impl
{
auto vectorAddImpl(const torch::Tensor& A,
                   const torch::Tensor& B) -> torch::Tensor;
auto cvtRGBtoGrayImpl(const torch::Tensor& img) -> torch::Tensor;
}  // namespace pmpp::ops::cuda::torch_impl
