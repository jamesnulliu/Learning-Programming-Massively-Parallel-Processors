#include <cuda_runtime.h>

#include "pmpp/types/cxx_types.hpp"
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
}
}  // namespace pmpp::ops::cuda