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
    uint32_t stride = blockDim.x;
    uint32_t segmentId = blockIdx.x;
    uint32_t segmentSize = 2 * stride;
    // Block thread index
    uint32_t bTidx = threadIdx.x;
    // Global thread index
    uint32_t gTidx = segmentId * segmentSize + bTidx;

    extern __shared__ ScalarT shmem[];

    shmem[bTidx] = pred(in[gTidx], in[gTidx + stride]);
    stride /= 2;

    for (; stride >= 1; stride /= 2) {
        __syncthreads();
        if (bTidx < stride) {
            shmem[bTidx] = pred(shmem[bTidx], shmem[bTidx + stride]);
        }
    }
    if (bTidx == 0) {
        atomicAdd(out, shmem[0]);
    }
}

template <typename ScalarT, typename PredT>
[[nodiscard]] auto launchReduction(const ScalarT* in, uint32_t n,
                                   const PredT& pred) -> ScalarT
{
    constexpr uint32_t MAX_BLOCK_THREADS = 1024;

    ScalarT* d_out;
    cudaMalloc(&d_out, 1 * sizeof(ScalarT));

    uint32_t stride = std::min(n / 2, MAX_BLOCK_THREADS);
    dim3 blockDim = {stride, 1, 1};
    dim3 gridDim = {ceilDiv(n, stride * 2), 1, 1};
    uint32_t shmemSize = stride * sizeof(ScalarT);

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