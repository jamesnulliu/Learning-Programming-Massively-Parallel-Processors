#pragma once

#include "pmpp/pch.hpp"

namespace pmpp::ops::cpu
{
void launchVecAdd(const fp32_t* a, const fp32_t* b, fp32_t* c, size_t n)
{
#pragma omp for
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

namespace torch_impl
{
inline auto vectorAdd(const torch::Tensor& A, const torch::Tensor& B) -> torch::Tensor
{
    torch::Tensor C = torch::zeros_like(A);

    switch (A.scalar_type()) {
    case torch::kFloat32: {
        pmpp::ops::cpu::launchVecAdd(
            A.data_ptr<fp32_t>(), B.data_ptr<fp32_t>(), C.data_ptr<fp32_t>(),
            A.flatten().size(0));
        break;
    }
    default:
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }

    return C;
}
}
}  // namespace pmpp::ops::cpu
