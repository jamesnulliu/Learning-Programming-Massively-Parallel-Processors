#pragma once

#include "pmpp/pch.hpp"
#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{
__global__ void vecAddKernel(const fp32_t* a, const fp32_t* b, fp32_t* c,
                             int32_t n)
{

    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid < n) {
        // [GM] 2 load, 1 store, 3 inst
        c[gtid] = a[gtid] + b[gtid];
    }
}

void launchVecAdd(const fp32_t* d_A, const fp32_t* d_B, fp32_t* d_C, size_t n)
{
    dim3 blockSize = {std::min<uint32_t>(n, 1024), 1, 1};
    dim3 gridSize = {ceilDiv<uint32_t>(n, blockSize.x), 1, 1};

    vecAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, int32_t(n));

    PMPP_DEBUG_CUDA_ERR_CHECK(cudaGetLastError());
}

namespace torch_impl
{
inline auto vectorAdd(const torch::Tensor& A, const torch::Tensor& B)
    -> torch::Tensor
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
}  // namespace torch_impl
}  // namespace pmpp::ops::cuda