#include <fmt/format.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "../torch_impl.hpp"
#include "pmpp/types/cxx_types.hpp"

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

namespace pmpp::ops::cpu
{
extern void launchVecAdd(const fp32_t* a, const fp32_t* b, fp32_t* c,
                         size_t n);

namespace torch_impl
{
auto vectorAddImpl(const torch::Tensor& A,
                   const torch::Tensor& B) -> torch::Tensor
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
}  // namespace torch_impl
}  // namespace pmpp::ops::cpu

namespace pmpp::ops::cuda
{
extern void launchVecAdd(const fp32_t* d_A, const fp32_t* d_B, fp32_t* d_C,
                         size_t n);

namespace torch_impl
{
auto vectorAddImpl(const torch::Tensor& A,
                   const torch::Tensor& B) -> torch::Tensor
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
}  // namespace torch_impl
}  // namespace pmpp::ops::cuda