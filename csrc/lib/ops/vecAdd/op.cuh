#pragma once

#include "pmpp/pch.hpp"
#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{
__global__ void vecAddKernelv0(const fp32_t* a, const fp32_t* b, fp32_t* c,
                               int32_t n)
{

    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid < n) {
        // [DRAM] 2 load, 1 store, 3 inst
        c[gtid] = a[gtid] + b[gtid];
    }
}

__global__ void vecAddKernelv1(const fp32_t* a, const fp32_t* b, fp32_t* c,
                               int32_t n)
{

    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    gtid = gtid % 2 == 0 ? gtid + 1 : gtid - 1;
    if (gtid < n) {
        // [DRAM] 2 load, 1 store, 3 inst
        c[gtid] = a[gtid] + b[gtid];
    }
}

__global__ void vecAddKernelv2(const fp32_t* a, const fp32_t* b, fp32_t* c,
                               int32_t n)
{

    int gtid = threadIdx.x + blockDim.x * blockIdx.x + 1;
    if (gtid < n) {
        // [DRAM] 2 load, 1 store, 3 inst
        c[gtid] = a[gtid] + b[gtid];
    }
}

template <uint8_t VERSION = 0>
void launchVecAdd(const fp32_t* d_A, const fp32_t* d_B, fp32_t* d_C, size_t n)
{
    dim3 blockSize = {std::min<uint32_t>(n, 1024), 1, 1};
    dim3 gridSize = {ceilDiv<uint32_t>(n, blockSize.x), 1, 1};

    if constexpr (VERSION == 0) {
        vecAddKernelv0<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    } else if (VERSION == 1) {
        vecAddKernelv1<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    } else if (VERSION == 2) {
        vecAddKernelv2<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    } else {
        PMPP_ABORT(std::format("Unsupported version: {}", VERSION).c_str());
    }

    PMPP_DEBUG_CUDA_ERR_CHECK(cudaGetLastError());
}

namespace torch_impl
{
template <uint8_t VERSION = 0>
inline auto vectorAdd(const torch::Tensor& A, const torch::Tensor& B)
    -> torch::Tensor
{
    torch::Tensor C = torch::empty_like(A);

    switch (A.scalar_type()) {
    case torch::kFloat32: {
        pmpp::ops::cuda::launchVecAdd<VERSION>(
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