#include <torch/torch.h>
#include <torch/types.h>

#include "pmpp/ops/vec_add.hpp"
#include "pmpp/types/cu_types.cuh"
#include "pmpp/types/cxx_types.hpp"
#include "pmpp/types/device.hpp"

namespace pmpp
{

template <typename ScalarT>
auto __vector_add_impl(const torch::Tensor& A, const torch::Tensor& B)
    -> torch::Tensor
{
    auto nElems = pmpp::size_t(A.size(0));
    auto C = torch::empty_like(A);

    // If A and B both on CUDA
    if (A.is_cuda() && B.is_cuda()) {
        pmpp::ops::launchVecAdd<ScalarT, pmpp::Device::CUDA>(
            A.data_ptr<ScalarT>(), B.data_ptr<ScalarT>(), C.data_ptr<ScalarT>(),
            nElems);
    } else if (A.is_cpu() && B.is_cpu()) {
        pmpp::ops::launchVecAdd<ScalarT, pmpp::Device::CPU>(
            A.data_ptr<ScalarT>(), B.data_ptr<ScalarT>(), C.data_ptr<ScalarT>(),
            nElems);
    }
    return C;
}

auto __vector_add(const torch::Tensor& A, const torch::Tensor& B)
    -> torch::Tensor
{
    // Check if tensors are on the same device
    TORCH_CHECK(A.device() == B.device(),
                "Tensors must be both on CPU or both on CUDA");

    // Check if tensors have the same dtype
    TORCH_CHECK(A.dtype() == B.dtype(),
                "Expected tensors to have the same dtype, but got ", A.dtype(),
                " and ", B.dtype());

    // Check if tensors have the same size
    TORCH_CHECK(A.sizes() == B.sizes(),
                "Expected tensors to have the same size, but got ", A.sizes(),
                " and ", B.sizes());

    switch (A.scalar_type()) {
    case torch::kF16: {
        // [TODO]
        // // return __vector_add_impl<pmpp::fp16_t>(A, B);
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }
    case torch::kF32: {
        return __vector_add_impl<pmpp::fp32_t>(A, B);
        break;
    }
    case torch::kF64: {
        // [TODO]
        // // return __vector_add_impl<pmpp::fp64_t>(A, B);
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }
    case torch::kI32: {
        // [TODO]
        // // return __vector_add_impl<pmpp::int32_t>(A, B);
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }
    case torch::kI64: {
        // [TODO]
        // // return __vector_add_impl<pmpp::int64_t>(A, B);
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }
    default:
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }
}

}  // namespace pmpp

// Define operator `torch.ops.pmpp.vector_add`.
// @see
// "https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.fu2gkc7w0nrc"
TORCH_LIBRARY(pmpp, m)
{
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
}

// Register the implementation.
// @see
// "https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.jc288bcufw9a"
TORCH_LIBRARY_IMPL(pmpp, CPU, m)
{
    m.impl("vector_add", &::pmpp::__vector_add);
}
TORCH_LIBRARY_IMPL(pmpp, CUDA, m)
{
    m.impl("vector_add", &::pmpp::__vector_add);
}