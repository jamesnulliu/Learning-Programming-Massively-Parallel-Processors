#include "torch/torch.h"

#include "../ops.hpp"
#include "../torch_impl.hpp"

namespace pmpp::ops::cpu::torch_impl
{
auto matmul(const torch::Tensor& A, const torch::Tensor& B) -> torch::Tensor
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
}  // namespace pmpp::ops::cpu::torch_impl

namespace pmpp::ops::cuda::torch_impl
{
auto matmul(const torch::Tensor& A, const torch::Tensor& B) -> torch::Tensor
{
    torch::Tensor C = torch::empty({A.size(0), B.size(1)}, A.options());

    switch (A.scalar_type()) {
    case torch::kFloat32: {
        pmpp::ops::cuda::launchMatmul(A.data_ptr<fp32_t>(),
                                      B.data_ptr<fp32_t>(),
                                      C.data_ptr<fp32_t>(), A.size(0));
        break;
    }
    default:
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }

    return C;
}
}  // namespace pmpp::ops::cuda::torch_impl