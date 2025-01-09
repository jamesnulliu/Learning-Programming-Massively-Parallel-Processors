#include <torch/torch.h>

#include "./torch_impl.hpp"

// Define operator `torch.ops.pmpp.vector_add`.
// @see
//   https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.fu2gkc7w0nrc
TORCH_LIBRARY(pmpp, m)
{
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
    m.def("cvt_rgb_to_gray(Tensor img) -> Tensor");
}

// Register the implementation.
// @see
//   https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.jc288bcufw9a
TORCH_LIBRARY_IMPL(pmpp, CPU, m)
{
    m.impl("vector_add", &pmpp::ops::cpu::torch_impl::vectorAddImpl);
    m.impl("cvt_rgb_to_gray", &pmpp::ops::cpu::torch_impl::cvtRGBtoGrayImpl);
}

TORCH_LIBRARY_IMPL(pmpp, CUDA, m)
{
    m.impl("vector_add", &pmpp::ops::cuda::torch_impl::vectorAddImpl);
    m.impl("cvt_rgb_to_gray", &pmpp::ops::cuda::torch_impl::cvtRGBtoGrayImpl);
}