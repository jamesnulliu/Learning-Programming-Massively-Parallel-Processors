#pragma once
#include "pmpp/pch.hpp"

#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{

template <typename ScalarT, typename PredT>
__global__ void reductionKernel(const ScalarT* in, uint32_t n, ScalarT* out,
                                const PredT& pred)
{
    // Thread index in the block
    uint32_t bTid = threadIdx.x;
    extern __shared__ ScalarT shmem[];

    uint32_t stride = blockDim.x;
    shmem[bTid] = pred(in[bTid], in[bTid + stride]);
    stride /= 2;

    for (; stride >= 1; stride /= 2) {
        __syncthreads();
        if (bTid < stride) {
            shmem[bTid] = pred(shmem[bTid], shmem[bTid + stride]);
        }
    }
    if (bTid == 0) {
        out[0] = shmem[0];
    }
}

template <typename ScalarT, typename PredT>
[[nodiscard]] auto launchReduction(const ScalarT* in, uint32_t n,
                                   const PredT& pred) -> ScalarT
{
    ScalarT* d_out;
    cudaMalloc(&d_out, 1 * sizeof(ScalarT));

    uint32_t nTreads = n / 2;
    dim3 blockDim = {nTreads, 1, 1};
    dim3 gridDim = {1, 1, 1};
    uint32_t shmemSize = blockDim.x * sizeof(ScalarT);

    reductionKernel<<<gridDim, blockDim, shmemSize>>>(in, n, d_out, pred);

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
    torch::Tensor result = {};

    switch (in.scalar_type()) {
    case torch::kFloat32: {
        result =
            torch::tensor(launchReduction(in.const_data_ptr<fp32_t>(),
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