#include <torch/torch.h>

#include "../ops.hpp"
#include "../torch_impl.hpp"

namespace pmpp::ops::cpu::torch_impl
{
auto alphabetHistogram(const torch::Tensor& input, int64_t divider)
    -> torch::Tensor
{
    auto nInputs = input.numel();
    auto histo = torch::zeros({26 / divider}, torch::kInt32);

    switch (input.scalar_type()) {
    case torch::kInt32: {
        pmpp::ops::cpu::launchAlphabetHistogram<int32_t>(
            input.data_ptr<int32_t>(), histo.data_ptr<int32_t>(), nInputs,
            int32_t(divider));
        break;
    }
    default: {
        AT_ERROR("Unsupported dtype: ", input.dtype());
    }
    }

    return histo;
}
}  // namespace pmpp::ops::cpu::torch_impl

namespace pmpp::ops::cuda::torch_impl
{
auto alphabetHistogram(const torch::Tensor& input, int64_t divider)
    -> torch::Tensor
{
    auto nInputs = input.numel();
    auto histo = torch::zeros({26 / divider}, torch::kInt32);

    switch (input.scalar_type()) {
    case torch::kInt32: {
        pmpp::ops::cuda::launchAlphabetHistogram<int32_t>(
            input.data_ptr<int32_t>(), histo.data_ptr<int32_t>(), nInputs,
            int32_t(divider));
        break;
    }
    default: {
        AT_ERROR("Unsupported dtype: ", input.dtype());
    }
    }

    return histo;
}
}  // namespace pmpp::ops::cuda::torch_impl