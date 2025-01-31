#include <cassert>
#include <cuda_runtime.h>

#include "../ops.hpp"
#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{
/**
 * @brief Matrix multiplication kernel
 *
 * @note 1. A, B, C are square matrices of size (m, m);
 *       2. Each thread computes COARSE_FACTOR element of C and each block
 *          computes (TILE_SIZE, TILE_SIZE) elements of C, which means block
 *          size should be (TILE_SIZE, TILE_SIZE);
 * @todo Add boundary checks.
 */
template <typename ScalarT, uint32_t TILE_SIZE = 32,
          uint32_t COARSE_FACTOR = 1>
__global__ void matmulKernel(const ScalarT* A, const ScalarT* B, ScalarT* C,
                             int32_t width)
{
    __shared__ ScalarT sA[TILE_SIZE][TILE_SIZE];
    __shared__ ScalarT sB[TILE_SIZE][TILE_SIZE];

    uint32_t row = blockIdx.x * TILE_SIZE + threadIdx.x;
    uint32_t colStart = blockIdx.y * TILE_SIZE * COARSE_FACTOR + threadIdx.y;

    ScalarT tmp[COARSE_FACTOR] = {};
    for (uint32_t i = 0; i < COARSE_FACTOR; ++i) {
        tmp[i] = 0;
    }

    for (uint32_t ti = 0; ti < width / TILE_SIZE; ++ti) {
        sA[threadIdx.x][threadIdx.y] =
            A[row * width + ti * TILE_SIZE + threadIdx.y];

        for (uint32_t c = 0; c < COARSE_FACTOR; ++c) {

            uint32_t col = colStart + c * TILE_SIZE;

            sB[threadIdx.x][threadIdx.y] =
                B[(ti * TILE_SIZE + threadIdx.x) * width + col];

            __syncthreads();

            for (uint32_t tii = 0; tii < TILE_SIZE; ++tii) {
                tmp[c] += sA[threadIdx.x][tii] * sB[tii][threadIdx.y];
            }
            __syncthreads();
        }
    }

    for (uint32_t c = 0; c < COARSE_FACTOR; ++c) {
        uint32_t col = colStart + c * TILE_SIZE;
        C[row * width + col] = tmp[c];
    }
}

void launchMatmul(const fp32_t* dA, const fp32_t* dB, fp32_t* dC, size_t width)
{
    constexpr uint32_t tileSize = 32;

    if (width % tileSize != 0) {
        throw std::runtime_error(
            "Matrix size should be multiple of tile size");
    }

    dim3 blockSize = {tileSize, tileSize};
    dim3 gridSize = {uint32_t(ceilDiv(width, tileSize)),
                     uint32_t(ceilDiv(width, tileSize))};

    matmulKernel<fp32_t, tileSize>
        <<<gridSize, blockSize>>>(dA, dB, dC, int32_t(width));

    PMPP_DEBUG_CUDA_ERR_CHECK(cudaGetLastError());
}
}  // namespace pmpp::ops::cuda