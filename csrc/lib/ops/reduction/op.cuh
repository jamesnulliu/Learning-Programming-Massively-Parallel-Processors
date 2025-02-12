#pragma once
#include "pmpp/pch.hpp"

#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{

template <typename ScalarT, typename PredT>
__global__ void reductionKernel(ScalarT* in, ScalarT* out, const PredT& pred)
{
    // Thread index in the block
    int32_t bTid = threadIdx.x;
    int32_t i = bTid * 2;
    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        if (bTid % stride == 0) {
            in[i] = pred(in[i], in[i + stride]);
        }
        __syncthreads();
    }
    if (bTid == 0) {
        out[blockIdx.x] = in[0];
    }
}

template <typename ScalarT, typename PredT>
[[nodiscard]] auto launchReduction(ScalarT* in, int32_t n, const PredT& pred)
    -> ScalarT
{
    dim3 blockDim = {uint32_t(n), 1, 1};
    dim3 gridDim = {uint32_t(ceilDiv(n, blockDim.x)), 1, 1};
    ScalarT* d_out;
    cudaMalloc(&d_out, gridDim.x * sizeof(ScalarT));
    reductionKernel<<<gridDim, blockDim>>>(in, d_out, pred);
    ScalarT out;
    cudaMemcpy(&out, d_out, sizeof(ScalarT), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    PMPP_DEBUG_CUDA_ERR_CHECK(cudaGetLastError());

    return out;
}

namespace torch_impl
{
[[nodiscard]] inline auto mulReduction(const torch::Tensor& in)
    -> torch::Tensor
{
    torch::Tensor mutableIn = in.contiguous();
    torch::Tensor result = {};

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
}  // namespace pmpp::ops::cuda