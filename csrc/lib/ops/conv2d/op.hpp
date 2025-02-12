#pragma once
#include "pmpp/pch.hpp"

namespace pmpp::ops::cpu
{
template <typename ScalarT>
void launchConv2d(const ScalarT* input, const ScalarT* kernel, ScalarT* output,
                  int32_t inHeight, int32_t inWidth, int32_t kernelSize)
{
    for (int32_t i = 0; i < inHeight; ++i) {
        for (int32_t j = 0; j < inWidth; ++j) {
            fp32_t tmp = 0;
            int32_t startRow = i - kernelSize / 2 < 0 ? 0 : i - kernelSize / 2;
            int32_t startCol = j - kernelSize / 2 < 0 ? 0 : j - kernelSize / 2;
            int32_t endRow = i + kernelSize / 2 >= inHeight
                                 ? inHeight - 1
                                 : i + kernelSize / 2;
            int32_t endCol = j + kernelSize / 2 >= inWidth
                                 ? inWidth - 1
                                 : j + kernelSize / 2;

            for (int32_t k = startRow; k <= endRow; ++k) {
                for (int32_t l = startCol; l <= endCol; ++l) {
                    tmp += input[k * inWidth + l] *
                           kernel[(k - i + kernelSize / 2) * kernelSize +
                                  (l - j + kernelSize / 2)];
                }
            }
            output[i * inWidth + j] = tmp;
        }
    }
}

namespace torch_impl
{
auto conv2d(const torch::Tensor& input, const torch::Tensor& kernel)
    -> torch::Tensor
{
    TORCH_CHECK(input.scalar_type() == kernel.scalar_type(),
                "Expected input and kernel to have the same dtype, but got "
                "input.dtype = ",
                torch::toString(input.scalar_type()),
                " and kernel.dtype = ", torch::toString(kernel.scalar_type()));

    auto input_height = input.size(0);
    auto input_width = input.size(1);
    auto kernel_size = kernel.size(0);

    torch::Tensor output = torch::zeros_like(input);

    switch (input.scalar_type()) {
    case torch::kFloat32: {
        pmpp::ops::cpu::launchConv2d(input.const_data_ptr<fp32_t>(),
                                     kernel.const_data_ptr<fp32_t>(),
                                     output.mutable_data_ptr<fp32_t>(),
                                     input_height, input_width, kernel_size);
        break;
    }
    default: {
        AT_ERROR("Unsupported dtype: ", input.dtype());
    }
    }

    return output;
}
}  // namespace torch_impl
}  // namespace pmpp::ops::cpu