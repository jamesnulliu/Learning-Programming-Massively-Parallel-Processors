#include <torch/torch.h>

#include "../ops.hpp"
#include "../torch_impl.hpp"

namespace pmpp::ops::cpu::torch_impl
{
auto conv2D(const torch::Tensor& input, const torch::Tensor& kernel)
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
        pmpp::ops::cpu::launchConv2D(input.const_data_ptr<fp32_t>(),
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
}  // namespace pmpp::ops::cpu::torch_impl

namespace pmpp::ops::cuda::torch_impl
{
auto conv2D(const torch::Tensor& input, const torch::Tensor& kernel)
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
        pmpp::ops::cuda::launchConv2D(
            input.data_ptr<fp32_t>(), kernel.data_ptr<fp32_t>(),
            output.data_ptr<fp32_t>(), input_height, input_width, kernel_size);
        break;
    }
    default: {
        AT_ERROR("Unsupported dtype: ", input.dtype());
    }
    }

    return output;
}
}  // namespace pmpp::ops::cuda::torch_impl
