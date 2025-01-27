#include <fmt/format.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "../ops.hpp"
#include "../torch_impl.hpp"

#define VECTOR_ADD_CHECK(A, B, _Device)                                       \
    do {                                                                      \
        TORCH_CHECK(A.device() == B.device(),                                 \
                    fmt::format("Tensors must be both on " _Device            \
                                ", but got {} and {}.",                       \
                                A.device().str(), B.device().str()));         \
        TORCH_CHECK(A.dtype() == B.dtype(),                                   \
                    fmt::format("Expected tensors to have the same dtype, "   \
                                "but got {} and {}.",                         \
                                A.dtype().name(), B.dtype().name()));         \
        TORCH_CHECK(A.sizes() == B.sizes(),                                   \
                    "Expected tensors to have the same size, but got ",       \
                    A.sizes(), " and ", B.sizes(), ".");                      \
    } while (false)

namespace pmpp::ops::cpu::torch_impl
{
auto vectorAdd(const torch::Tensor& A, const torch::Tensor& B) -> torch::Tensor
{
    VECTOR_ADD_CHECK(A, B, "CPU");

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
}  // namespace pmpp::ops::cpu::torch_impl

namespace pmpp::ops::cuda::torch_impl
{
auto vectorAdd(const torch::Tensor& A, const torch::Tensor& B) -> torch::Tensor
{
    VECTOR_ADD_CHECK(A, B, "CUDA");

    torch::Tensor C = torch::empty_like(A);

    switch (A.scalar_type()) {
    case torch::kFloat32: {
        pmpp::ops::cuda::launchVecAdd(
            A.data_ptr<fp32_t>(), B.data_ptr<fp32_t>(), C.data_ptr<fp32_t>(),
            A.flatten().size(0));
        break;
    }
    default:
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }

    return C;
}
}  // namespace pmpp::ops::cuda::torch_impl
