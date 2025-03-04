#pragma once

#include "pmpp/pch.hpp"

#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{

template <typename ScalarT>
__global__ void koggeStonePrefixSumKernel(const ScalarT* input,
                                          ScalarT* output, uint32_t n)
{
    extern __shared__ ScalarT shmem[];

    uint32_t btid = threadIdx.x;                            // Block Thread ID
    uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;  // Global Thread ID

    if (gtid < n) {
        shmem[btid] = input[gtid];
    } else {
        shmem[btid] = 0;
    }

    __syncthreads();

    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        ScalarT tmp = 0;
        if (btid >= stride) {
            tmp = shmem[btid] + shmem[btid - stride];
        }
        __syncthreads();
        if (btid >= stride) {
            shmem[btid] = tmp;
        }
        __syncthreads();
    }

    if (gtid < n) {
        output[gtid] = shmem[btid];
    }
}

template <typename ScalarT>
void launchPrefixSum(const ScalarT* d_input, ScalarT* d_output, uint32_t n)
{
    constexpr uint32_t blockSize = 256;
    uint32_t gridSize = ceilDiv(n, blockSize);
    koggeStonePrefixSumKernel<<<gridSize, blockSize,
                                blockSize * sizeof(ScalarT)>>>(d_input,
                                                               d_output, n);
}

namespace torch_impl
{
inline auto prefixSum(const torch::Tensor& A) -> torch::Tensor
{
    torch::Tensor B = torch::empty_like(A);
    switch (A.scalar_type()) {
    case torch::kFloat32: {
        pmpp::ops::cuda::launchPrefixSum(A.data_ptr<fp32_t>(),
                                         B.data_ptr<fp32_t>(), A.size(0));
        break;
    }
    default:
        AT_ERROR("Unsupported dtype: ", A.dtype());
    }

    return B;
}
}  // namespace torch_impl

}  // namespace pmpp::ops::cuda