#include <cuda_runtime.h>

#include "../ops.hpp"
#include "pmpp/utils/address.hpp"
#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{

template <typename ScalarT>
__global__ void conv2DKernel(const ScalarT* input, const ScalarT* kernel,
                             ScalarT* output, int32_t input_height,
                             int32_t input_width, int32_t kernel_size)
{
    int32_t outRow = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t outCol = blockIdx.y * blockDim.y + threadIdx.y;

    ScalarT tmp = 0;
    for (int32_t kRow = 0; kRow < kernel_size; ++kRow) {
        for (int32_t kCol = 0; kCol < kernel_size; ++kCol) {
            int32_t inRow = outRow + kRow - kernel_size / 2;
            int32_t inCol = outCol + kCol - kernel_size / 2;
            if (inRow >= 0 && inRow < input_height && inCol >= 0 &&
                inCol < input_width) {
                tmp += input[computeOffset<int32_t>(inRow, inCol, input_width,
                                                    input_width)] *
                       kernel[computeOffset<int32_t>(kRow, kCol, kernel_size,
                                                     kernel_size)];
            }
        }
    }
    output[computeOffset<int32_t>(outRow, outCol, input_width, input_width)] =
        tmp;
}

template <>
void launchConv2D<fp32_t>(const fp32_t* d_input, const fp32_t* d_kernel,
                          fp32_t* d_output, int32_t inputHeight,
                          int32_t inputWidth, int32_t kernelSize)
{
    dim3 blockSize = {32, 32, 1};
    dim3 gridSize = {uint32_t(ceilDiv(inputWidth, blockSize.x)),
                     uint32_t(ceilDiv(inputHeight, blockSize.y))};
    conv2DKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output,
                                          inputHeight, inputWidth, kernelSize);

    PMPP_DEBUG_CUDA_ERR_CHECK(cudaGetLastError());
}
}  // namespace pmpp::ops::cuda