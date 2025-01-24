#include <cuda_runtime.h>

#include "pmpp/types/cxx_types.hpp"

namespace pmpp::ops::cuda
{
/**
 * Assumes:
 * 1. M, N, P are square matrices of size width x width;
 * 2. Each thread computes one element;
 */
template <int32_t TILE_SIZE = 16, typename ScalarT = fp32_t>
__global__ void matMulKernel(ScalarT* M, ScalarT* N, ScalarT* P, int32_t Width)
{
    __shared__ ScalarT Mds[TILE_SIZE][TILE_SIZE];
    __shared__ ScalarT Nds[TILE_SIZE][TILE_SIZE];

    int32_t Row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int32_t Col = blockIdx.x * TILE_SIZE + threadIdx.x;

    fp32_t Pvalue = 0.0F;
    for (int32_t ph = 0; ph < Width / TILE_SIZE; ++ph) {
        Mds[threadIdx.y][threadIdx.x] = M[Row * Width + (ph * TILE_SIZE + threadIdx.x)];
        Nds[threadIdx.y][threadIdx.x] = N[(ph * TILE_SIZE + threadIdx.y) * Width + Col];
        __syncthreads();

        for (int32_t k = 0; k < TILE_SIZE; ++k) {
            Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        }
        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}
}  // namespace pmpp::ops::cuda