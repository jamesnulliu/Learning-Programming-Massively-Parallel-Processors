#pragma once

#include "pmpp/pch.hpp"

namespace pmpp::ops::cpu
{
template <typename ScalarT>
void launchPrefixSum(ScalarT* input, ScalarT* output, size_t n)
{
    output[0] = input[0];
    for (size_t i = 1; i < n; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

namespace torch_impl
{
inline auto prefixSum(const torch::Tensor& A) -> torch::Tensor
{
    torch::Tensor B = torch::empty_like(A);
    switch (A.scalar_type()) {
    case torch::kFloat32: {
        pmpp::ops::cpu::launchPrefixSum(A.data_ptr<fp32_t>(),
                                        B.data_ptr<fp32_t>(), A.size(0));
        break;
    }
    case torch::kInt32: {
        pmpp::ops::cpu::launchPrefixSum(A.data_ptr<int32_t>(),
                                        B.data_ptr<int32_t>(), A.size(0));
        break;
    }
    default:
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }

    return B;
} 
}  // namespace torch_impl
}  // namespace pmpp::ops::cpu
