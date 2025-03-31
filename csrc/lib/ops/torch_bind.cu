#include "pmpp/pch.hpp"

#include "./torch_impl.hpp"

// Define operator `torch.ops.pmpp.vector_add`.
// @see
//   https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.fu2gkc7w0nrc
TORCH_LIBRARY(pmpp, m)
{
    m.def("vector_add_v0(Tensor a, Tensor b) -> Tensor");
    m.def("vector_add_v1(Tensor a, Tensor b) -> Tensor");
    m.def("vector_add_v2(Tensor a, Tensor b) -> Tensor");
    m.def("cvt_rgb_to_gray(Tensor img) -> Tensor");
    m.def("matmul(Tensor A, Tensor B) -> Tensor");
    m.def("conv2d(Tensor input, Tensor kernel) -> Tensor");
    m.def("alphabet_histogram(Tensor input, int divider) -> Tensor");
    m.def("mul_reduction(Tensor input) -> Tensor");
    m.def("prefix_sum(Tensor input) -> Tensor");
}

// Register the implementations.
// @see
//   https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.jc288bcufw9a
TORCH_LIBRARY_IMPL(pmpp, CPU, m)
{
    m.impl("vector_add_v0", &pmpp::ops::cpu::torch_impl::vectorAdd);
    m.impl("vector_add_v1", &pmpp::ops::cpu::torch_impl::vectorAdd);
    m.impl("vector_add_v2", &pmpp::ops::cpu::torch_impl::vectorAdd);
    m.impl("cvt_rgb_to_gray", &pmpp::ops::cpu::torch_impl::cvtRGBtoGray);
    m.impl("matmul", &pmpp::ops::cpu::torch_impl::matmul);
    m.impl("conv2d", &pmpp::ops::cpu::torch_impl::conv2d);
    m.impl("alphabet_histogram",
           &pmpp::ops::cpu::torch_impl::alphabetHistogram);
    m.impl("mul_reduction", &pmpp::ops::cpu::torch_impl::mulReduction);
    m.impl("prefix_sum", &pmpp::ops::cpu::torch_impl::prefixSum);
}

TORCH_LIBRARY_IMPL(pmpp, CUDA, m)
{
    m.impl("vector_add_v0", &pmpp::ops::cuda::torch_impl::vectorAdd<0>);
    m.impl("vector_add_v1", &pmpp::ops::cuda::torch_impl::vectorAdd<1>);
    m.impl("vector_add_v2", &pmpp::ops::cuda::torch_impl::vectorAdd<2>);
    m.impl("cvt_rgb_to_gray", &pmpp::ops::cuda::torch_impl::cvtRGBtoGray);
    m.impl("matmul", &pmpp::ops::cuda::torch_impl::matmul);
    m.impl("conv2d", &pmpp::ops::cuda::torch_impl::conv2d);
    m.impl("alphabet_histogram",
           &pmpp::ops::cuda::torch_impl::alphabetHistogram);
    m.impl("mul_reduction", &pmpp::ops::cuda::torch_impl::mulReduction);
    m.impl("prefix_sum", &pmpp::ops::cuda::torch_impl::prefixSum);
}