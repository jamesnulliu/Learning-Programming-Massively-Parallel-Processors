#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cuda_runtime.h>

#include "pmpp/ops/vec_add.hpp"
#include "pmpp/types/cxx_types.hpp"

namespace pmpp::ops
{

__global__ void vecAddKernel(const fp32_t* a, const fp32_t* b, fp32_t* c,
                             int32_t n)
{

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

template <>
void launchVecAdd<fp32_t, Device::CUDA>(const fp32_t* d_A, const fp32_t* d_B,
                                        fp32_t* d_C, size_t n)
{
    uint32_t block_size = 256;
    uint32_t grid_size = (n + block_size - 1) / block_size;


    // Apply Device Memory
    // float *d_A, *d_B, *d_C;
    // CUDA_ERR_CHECK(cudaMalloc((void**) &d_A, n * sizeof(fp32_t)));
    // CUDA_ERR_CHECK(cudaMalloc((void**) &d_B, n * sizeof(fp32_t)));
    // CUDA_ERR_CHECK(cudaMalloc((void**) &d_C, n * sizeof(fp32_t)));

    // // Copy Host Memory to Device Memory
    // CUDA_ERR_CHECK(
    //     cudaMemcpy(d_A, h_A, n * sizeof(fp32_t), cudaMemcpyHostToDevice));
    // CUDA_ERR_CHECK(
    //     cudaMemcpy(d_B, h_B, n * sizeof(fp32_t), cudaMemcpyHostToDevice));

    // Launch Kernel
    vecAddKernel<<<grid_size, block_size>>>(d_A, d_B, d_C, int32_t(n));

    // // Copy Device Memory to Host Memory
    // CUDA_ERR_CHECK(
    //     cudaMemcpy(h_C, d_C, n * sizeof(fp32_t), cudaMemcpyDeviceToHost));

    // // Free Device Memory
    // CUDA_ERR_CHECK(cudaFree(d_A));
    // CUDA_ERR_CHECK(cudaFree(d_B));
    // CUDA_ERR_CHECK(cudaFree(d_C));
}

}  // namespace pmpp::ops