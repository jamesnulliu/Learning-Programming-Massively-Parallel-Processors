#pragma once

#include <torch/torch.h>

namespace pmpp::ops::cpu::torch_impl
{

auto vectorAdd(const torch::Tensor& A, const torch::Tensor& B)
    -> torch::Tensor;

auto cvtRGBtoGray(const torch::Tensor& img) -> torch::Tensor;

auto matmul(const torch::Tensor& A, const torch::Tensor& B) -> torch::Tensor;

}  // namespace pmpp::ops::cpu::torch_impl

namespace pmpp::ops::cuda::torch_impl
{

auto vectorAdd(const torch::Tensor& A, const torch::Tensor& B)
    -> torch::Tensor;

auto cvtRGBtoGray(const torch::Tensor& img) -> torch::Tensor;

auto matmul(const torch::Tensor& A, const torch::Tensor& B) -> torch::Tensor;

}  // namespace pmpp::ops::cuda::torch_impl
