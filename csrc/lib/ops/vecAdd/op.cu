#include <cuda_runtime.h>

#include "pmpp/ops/vec_add.hpp"
#include "pmpp/utils/math.hpp"

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
void launchVecAdd<DeviceType::CUDA>(const fp32_t* d_A, const fp32_t* d_B,
                                    fp32_t* d_C, size_t n)
{
    dim3 blockSize = 256;
    dim3 gridSize = ceil(n, 256);

    vecAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, int32_t(n));
}

}  // namespace pmpp::ops