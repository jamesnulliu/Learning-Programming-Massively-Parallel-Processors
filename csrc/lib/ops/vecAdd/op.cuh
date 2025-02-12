#pragma once

#include "pmpp/pch.hpp"
#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{
__global__ void vecAddKernel(const fp32_t* a, const fp32_t* b, fp32_t* c,
                             int32_t n)
{

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void launchVecAdd(const fp32_t* d_A, const fp32_t* d_B, fp32_t* d_C, size_t n)
{
    dim3 blockSize = 256;
    dim3 gridSize = ceilDiv(n, 256);

    vecAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, int32_t(n));

    PMPP_DEBUG_CUDA_ERR_CHECK(cudaGetLastError());
}

namespace torch_impl
{
inline auto vectorAdd(const torch::Tensor& A, const torch::Tensor& B) -> torch::Tensor
{
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
}
}  // namespace pmpp::ops::cuda