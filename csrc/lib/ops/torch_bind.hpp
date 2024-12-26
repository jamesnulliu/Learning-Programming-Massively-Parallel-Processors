#pragma once

#include <torch/torch.h>

namespace pmpp::ops::torch_impl
{
auto vectorAddCpuImpl(const torch::Tensor& A,
                      const torch::Tensor& B) -> torch::Tensor;

auto vectorAddCudaImpl(const torch::Tensor& A,
                       const torch::Tensor& B) -> torch::Tensor;

auto cvtRGBtoGrayCpuImpl(const torch::Tensor& picIn) -> torch::Tensor;

auto cvtRGBtoGrayCudaImpl(const torch::Tensor& picIn) -> torch::Tensor;
}  // namespace pmpp::ops::torch_impl
