#include <cuda_runtime.h>

#include "../ops.hpp"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{
/**
 * @brief Matrix multiplication kernel
 *
 * @note 1. A, B, C are square matrices of size (m, m);
 *       2. Each thread computes 1 element of C and each block computes
 *          (TILE_SIZE, TILE_SIZE) elements of C, which means block size should
 *          be (TILE_SIZE, TILE_SIZE);
 * @todo Add boundary checks.
 */
template <int32_t TILE_SIZE = 32, typename ScalarT = fp32_t>
__global__ void matmulKernel(const ScalarT* A, const ScalarT* B, ScalarT* C,
                             int32_t m)
{
    __shared__ ScalarT Mds[TILE_SIZE][TILE_SIZE];
    __shared__ ScalarT Nds[TILE_SIZE][TILE_SIZE];

    int32_t row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int32_t col = blockIdx.y * TILE_SIZE + threadIdx.y;

    ScalarT tmp = 0.0F;
    for (int32_t ph = 0; ph < m / TILE_SIZE; ++ph) {
        Mds[threadIdx.x][threadIdx.y] =
            A[row * m + (ph * TILE_SIZE + threadIdx.y)];
        Nds[threadIdx.x][threadIdx.y] =
            B[(ph * TILE_SIZE + threadIdx.x) * m + col];
        __syncthreads();

        for (int32_t k = 0; k < TILE_SIZE; ++k) {
            tmp += Mds[threadIdx.x][k] * Nds[k][threadIdx.y];
        }
        __syncthreads();
    }

    C[row * m + col] = tmp;
}

void launchMatmul(const fp32_t* dA, const fp32_t* dB, fp32_t* dC, size_t m)
{
    constexpr uint32_t tileSize = 32;

    dim3 blockSize = {tileSize, tileSize};
    dim3 gridSize = {uint32_t(ceilDiv(m, tileSize)),
                     uint32_t(ceilDiv(m, tileSize))};

    matmulKernel<tileSize, fp32_t>
        <<<gridSize, blockSize>>>(dA, dB, dC, int32_t(m));
}
}  // namespace pmpp::ops::cuda