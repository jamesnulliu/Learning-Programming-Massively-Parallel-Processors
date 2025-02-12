#pragma once
#include "pmpp/pch.hpp"

#include "pmpp/utils/address.hpp"
#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{
static constexpr int32_t MAX_CONV2D_KERNEL_SIZE = 9;
static __constant__ fp32_t
        KERNEL[MAX_CONV2D_KERNEL_SIZE * MAX_CONV2D_KERNEL_SIZE];

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

template <typename ScalarT>
void launchConv2d(const ScalarT* d_input, const ScalarT* d_kernel,
                  ScalarT* d_output, int32_t inputHeight, int32_t inputWidth,
                  int32_t kernelSize)
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

namespace torch_impl
{
inline auto conv2d(const torch::Tensor& input, const torch::Tensor& kernel)
    -> torch::Tensor
{
    TORCH_CHECK(input.scalar_type() == kernel.scalar_type(),
                "Expected input and kernel to have the same dtype, but got "
                "input.dtype = ",
                torch::toString(input.scalar_type()),
                " and kernel.dtype = ", torch::toString(kernel.scalar_type()));

    auto input_height = input.size(0);
    auto input_width = input.size(1);
    auto kernel_size = kernel.size(0);

    torch::Tensor output = torch::zeros_like(input);

    switch (input.scalar_type()) {
    case torch::kFloat32: {
        pmpp::ops::cuda::launchConv2d(
            input.data_ptr<fp32_t>(), kernel.data_ptr<fp32_t>(),
            output.data_ptr<fp32_t>(), input_height, input_width, kernel_size);
        break;
    }
    default: {
        AT_ERROR("Unsupported dtype: ", input.dtype());
    }
    }

    return output;
}
}  // namespace torch_impl
}  // namespace pmpp::ops::cuda