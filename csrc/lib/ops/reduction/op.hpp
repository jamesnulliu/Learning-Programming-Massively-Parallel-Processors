#pragma once

#include "pmpp/pch.hpp"

namespace pmpp::ops::cpu
{
template <typename ScalarT, typename PredT>
auto launchReduction(const ScalarT* in, int32_t n, const PredT& pred)
    -> ScalarT
{
    ScalarT result = in[0];
    for (int32_t i = 1; i < n; ++i) {
        result = pred(result, in[i]);
    }
    return result;
}

namespace torch_impl
{
inline auto mulReduction(const torch::Tensor& in) -> torch::Tensor
{
    torch::Tensor mutableIn = in.contiguous();
    torch::Tensor result;

    switch (in.scalar_type()) {
    case torch::kFloat32: {
        result =
            torch::tensor(launchReduction(mutableIn.mutable_data_ptr<fp32_t>(),
                                          in.numel(), std::multiplies<>()),
                          in.options());
        break;
    }
    default: {
        TORCH_CHECK(false, "[libpmpp] Unsupported data type");
    }
    }
    return result;
}
}  // namespace torch_impl
}  // namespace pmpp::ops::cpu