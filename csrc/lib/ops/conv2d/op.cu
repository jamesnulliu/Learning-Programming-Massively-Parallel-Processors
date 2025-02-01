#include <cuda_runtime.h>

#include "../ops.hpp"
#include "pmpp/utils/address.hpp"
#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{

constexpr int32_t MAX_CONV2D_KERNEL_SIZE = 9;
__constant__ fp32_t
    CONV2D_KERNEL[MAX_CONV2D_KERNEL_SIZE * MAX_CONV2D_KERNEL_SIZE];

template <typename ScalarT, uint32_t IN_TILE_SIZE = 32>
__global__ void conv2DKernel(const ScalarT* input, const ScalarT* kernel,
                             ScalarT* output, int32_t inHeight,
                             int32_t inWidth, int32_t kernelSize)
{
    uint32_t OUT_TILE_SIZE = IN_TILE_SIZE - kernelSize / 2 * 2;

    int32_t outRow = blockIdx.x * OUT_TILE_SIZE + threadIdx.x - kernelSize / 2;
    int32_t outCol = blockIdx.y * OUT_TILE_SIZE + threadIdx.y - kernelSize / 2;

    // [NOTE] IN_TILE_SIZE must equal to blockDim.x and blockDim.y
    __shared__ ScalarT inTile[IN_TILE_SIZE][IN_TILE_SIZE];

    if (outRow >= 0 && outRow < inHeight && outCol >= 0 && outCol < inWidth) {
        inTile[threadIdx.x][threadIdx.y] =
            input[computeOffset<uint32_t>(outRow, outCol, inWidth, inHeight)];
    } else {
        inTile[threadIdx.x][threadIdx.y] = 0.0;
    }
    __syncthreads();

    int32_t outTileRow = threadIdx.x - kernelSize / 2;
    int32_t outTileCol = threadIdx.y - kernelSize / 2;

    if (outRow >= 0 && outRow < inHeight && outCol >= 0 && outCol < inWidth) {
        if (outTileRow >= 0 && outTileRow < OUT_TILE_SIZE && outTileCol >= 0 &&
            outTileCol < OUT_TILE_SIZE) {
            ScalarT tmp = 0;
            for (int32_t kRow = 0; kRow < kernelSize; ++kRow) {
                for (int32_t kCol = 0; kCol < kernelSize; ++kCol) {
                    tmp += CONV2D_KERNEL[computeOffset<uint32_t>(
                               kRow, kCol, kernelSize, kernelSize)] *
                           inTile[kRow + outTileRow][kCol + outTileCol];
                }
            }
            output[computeOffset<uint32_t>(outRow, outCol, inWidth, inWidth)] =
                tmp;
        }
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

    cudaMemcpyToSymbol(CONV2D_KERNEL, d_kernel,
                       kernelSize * kernelSize * sizeof(fp32_t));

    dim3 blockDim = {32, 32, 1};
    dim3 gridDim = {uint32_t(ceilDiv(inputWidth, blockDim.x)),
                    uint32_t(ceilDiv(inputHeight, blockDim.y))};
    conv2DKernel<fp32_t, 32><<<gridDim, blockDim>>>(
        d_input, d_kernel, d_output, inputHeight, inputWidth, kernelSize);

    PMPP_DEBUG_CUDA_ERR_CHECK(cudaGetLastError());
}
}  // namespace pmpp::ops::cuda