#pragma once

#include "pmpp/pch.hpp"

namespace pmpp::ops::cpu
{
void launchMatmul(const fp32_t* A, const fp32_t* B, fp32_t* C, size_t m)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            C[i * m + j] = 0;
            for (size_t k = 0; k < m; ++k) {
                C[i * m + j] += A[i * m + k] * B[k * m + j];
            }
        }
    }
}

namespace torch_impl
{
inline auto matmul(const torch::Tensor& A, const torch::Tensor& B)
    -> torch::Tensor
{
    torch::Tensor C = torch::empty({A.size(0), B.size(1)}, A.options());

    switch (A.scalar_type()) {
    case torch::kFloat32: {
        pmpp::ops::cpu::launchMatmul(A.data_ptr<fp32_t>(),
                                     B.data_ptr<fp32_t>(),
                                     C.data_ptr<fp32_t>(), A.size(0));
        break;
    }
    default:
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }

    return C;
}
}  // namespace torch_impl
}  // namespace pmpp::ops::cpu