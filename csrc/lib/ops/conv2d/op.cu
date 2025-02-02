#include <cuda_runtime.h>

#include "../ops.hpp"
#include "pmpp/utils/address.hpp"
#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{

constexpr int32_t MAX_CONV2D_KERNEL_SIZE = 9;
__constant__ fp32_t KERNEL[MAX_CONV2D_KERNEL_SIZE * MAX_CONV2D_KERNEL_SIZE];

template <typename ScalarT, uint32_t TILE_SIZE = 32>
__global__ void conv2DKernel(const ScalarT* input, const ScalarT* kernel,
                             ScalarT* output, int32_t nRows, int32_t nCols,
                             int32_t kernelSize)
{
    // Each block computes (TILE_SIZE, TILE_SIZE) output elements
    // Each block contains (TILE_SIZE, TILE_SIZE) threads
    // TILE_SIZE must equal to blockDim.x and blockDim.y

    // Current thread computes element at output[outRow, outCol]
    int32_t outRow = blockIdx.x * TILE_SIZE + threadIdx.x;
    int32_t outCol = blockIdx.y * TILE_SIZE + threadIdx.y;

    __shared__ ScalarT inTile[TILE_SIZE][TILE_SIZE];
    // Load input tile into shared memory
    if (outRow < nRows && outCol < nCols) {
        inTile[threadIdx.x][threadIdx.y] =
            input[offset<uint32_t>(outRow, outCol, nRows, nCols)];
    } else {
        inTile[threadIdx.x][threadIdx.y] = 0.0;
    }
    __syncthreads();

    if (outRow < nRows && outCol < nCols) {
        ScalarT tmp = 0;
        // To compute one output element, each thread needs to loop over the
        // kernel:
        for (int32_t kRow = 0; kRow < kernelSize; ++kRow) {
            for (int32_t kCol = 0; kCol < kernelSize; ++kCol) {
                // Realative kernel index in the input tile
                int32_t rkInRow = threadIdx.x - kernelSize / 2 + kRow;
                int32_t rkInCol = threadIdx.y - kernelSize / 2 + kCol;
                if (rkInRow >= 0 && rkInRow < TILE_SIZE && rkInCol >= 0 &&
                    rkInCol < TILE_SIZE) {
                    tmp += inTile[rkInRow][rkInCol] *
                           KERNEL[offset<uint32_t>(kRow, kCol, kernelSize,
                                                   kernelSize)];
                } else {
                    // Boundary
                    int32_t inRow = outRow - kernelSize / 2 + kRow;
                    int32_t inCol = outCol - kernelSize / 2 + kCol;
                    if (inRow >= 0 && inRow < nRows && inCol >= 0 &&
                        inCol < nCols) {
                        tmp += input[offset<uint32_t>(inRow, inCol, nRows,
                                                      nCols)] *
                               KERNEL[offset<uint32_t>(kRow, kCol, kernelSize,
                                                       kernelSize)];
                    }
                }
            }
        }
        output[offset<uint32_t>(outRow, outCol, nRows, nCols)] = tmp;
    }
}

template <>
void launchConv2d<fp32_t>(const fp32_t* d_input, const fp32_t* d_kernel,
                          fp32_t* d_output, int32_t inputHeight,
                          int32_t inputWidth, int32_t kernelSize)
{
    if (kernelSize > MAX_CONV2D_KERNEL_SIZE) {
        throw std::runtime_error("Kernel size is too large");
    }

    cudaMemcpyToSymbol(KERNEL, d_kernel,
                       kernelSize * kernelSize * sizeof(fp32_t));

    dim3 blockDim = {32, 32, 1};
    dim3 gridDim = {uint32_t(ceilDiv(inputWidth, blockDim.x)),
                    uint32_t(ceilDiv(inputHeight, blockDim.y))};
    conv2DKernel<fp32_t, 32><<<gridDim, blockDim>>>(
        d_input, d_kernel, d_output, inputHeight, inputWidth, kernelSize);

    PMPP_DEBUG_CUDA_ERR_CHECK(cudaGetLastError());
}
}  // namespace pmpp::ops::cuda