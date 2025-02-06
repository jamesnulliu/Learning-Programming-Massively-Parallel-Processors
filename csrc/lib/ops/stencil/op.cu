#include <cuda_runtime.h>

#include "../ops.hpp"
#include "pmpp/utils/address.hpp"
#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{

template <typename ScalarT, dim3 TILE_DIM>
__global__ void stencilKernel(const ScalarT* input, ScalarT* output,
                              dim3 shape, const std::array<ScalarT, 7>& coeffs)
{
    int32_t iStart = blockIdx.z * TILE_DIM.z;
    int32_t j = blockIdx.y * TILE_DIM.y + threadIdx.y - 1;
    int32_t k = blockIdx.x * TILE_DIM.x + threadIdx.x - 1;

    __shared__ ScalarT inPrev_s[TILE_DIM.x][TILE_DIM.y];
    __shared__ ScalarT inCurr_s[TILE_DIM.x][TILE_DIM.y];
    __shared__ ScalarT inNext_s[TILE_DIM.x][TILE_DIM.y];

    if (iStart - 1 >= 0 && iStart - 1 < shape.z && j >= 0 && j < shape.y &&
        k >= 0 && k < shape.x) {
        inPrev_s[threadIdx.y][threadIdx.x] = input[offset<uint32_t>(
            iStart - 1, j, k, shape.z, shape.y, shape.x)];
    }

    if (iStart >= 0 && iStart < shape.z && j >= 0 && j < shape.y && k >= 0 &&
        k < shape.x) {
        inCurr_s[threadIdx.y][threadIdx.x] =
            input[offset<uint32_t>(iStart, j, k, shape.z, shape.y, shape.x)];
    }

    for (int32_t i = iStart; i < iStart + TILE_DIM.z; ++i) {
        if (i + 1 >= 0 && i + 1 < shape.z && j >= 0 && j < shape.y && k >= 0 &&
            k < shape.x) {
            inNext_s[threadIdx.y][threadIdx.x] = input[offset<uint32_t>(
                i + 1, j, k, shape.z, shape.y, shape.x)];
        }
        __syncthreads();
        if (i >= 1 && i < shape.z - 1 && j >= 1 && j < shape.y - 1 && k >= 1 &&
            k < shape.x - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < TILE_DIM.y - 1 &&
                threadIdx.x >= 1 && threadIdx.x < TILE_DIM.x - 1) {
                output[offset<uint32_t>(i, j, k, shape.z, shape.y, shape.x)] =
                    coeffs[0] * inCurr_s[threadIdx.y][threadIdx.x] +
                    coeffs[1] * inCurr_s[threadIdx.y][threadIdx.x - 1] +
                    coeffs[2] * inCurr_s[threadIdx.y][threadIdx.x + 1] +
                    coeffs[3] * inCurr_s[threadIdx.y - 1][threadIdx.x] +
                    coeffs[4] * inCurr_s[threadIdx.y + 1][threadIdx.x] +
                    coeffs[5] * inPrev_s[threadIdx.y][threadIdx.x] +
                    coeffs[6] * inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] =
            inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] =
            inNext_s[threadIdx.y][threadIdx.x];
    }
}

template <>
void launchStencil3d(const fp32_t* input, fp32_t* output, dim3 shape,
                     const std::array<fp32_t, 7>& coeffs)
{
    constexpr dim3 BLOCK_DIM = {8, 8, 8};
    dim3 gridDim = {ceilDiv(shape.x, BLOCK_DIM.x),
                    ceilDiv(shape.y, BLOCK_DIM.y),
                    ceilDiv(shape.z, BLOCK_DIM.z)};
    
    

    stencilKernel<fp32_t, BLOCK_DIM>
        <<<gridDim, BLOCK_DIM>>>(input, output, shape, coeffs);

    PMPP_DEBUG_CUDA_ERR_CHECK(cudaGetLastError());
}

}  // namespace pmpp::ops::cuda