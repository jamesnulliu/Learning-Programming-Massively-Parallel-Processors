#pragma once
#include "pmpp/pch.hpp"

#include "pmpp/utils/common.cuh"
#include "pmpp/utils/math.hpp"

namespace pmpp::ops::cuda
{
template <typename ScalarT>
    requires std::is_integral_v<ScalarT>
__global__ void alphabetHistogramKernel(const ScalarT* input, ScalarT* histo,
                                        int32_t nInputs, int32_t divider)
{
    constexpr auto MAX_N_BINS = 26;
    int32_t nBins = ceilDiv(26, divider);
    __shared__ int32_t histo_s[MAX_N_BINS];
    ::pmpp::cuda::initMemory(histo_s, nBins, 0);
    __syncthreads();

    // Global thread index
    int32_t gTid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t accumulator = 0;
    int32_t prevBinIdx = -1;

    // Map concecutive threads to all elements of the input
    for (int32_t i = gTid; i < nInputs; i += blockDim.x * gridDim.x) {
        int32_t alphabetPos = input[i] - 'a';
        if (alphabetPos >= 0 && alphabetPos < 26) {
            int32_t bin = alphabetPos / divider;
            if (bin == prevBinIdx) {
                ++accumulator;
            } else {
                if (accumulator >= 0) {
                    atomicAdd(&(histo_s[prevBinIdx]), accumulator);
                }
                accumulator = 1;
                prevBinIdx = bin;
            }
        }
    }
    if (accumulator > 0) {
        atomicAdd(&(histo_s[prevBinIdx]), accumulator);
    }

    if (blockIdx.x > 0) {
        __syncthreads();
        // This loop is for the case when nBins > blockDim.x (nThreads per
        // block)
        for (int32_t bin = threadIdx.x; bin < nBins; bin += blockDim.x) {
            int32_t binVal = histo_s[bin];
            if (binVal > 0) {
                atomicAdd(&(histo[bin]), binVal);
            }
        }
    }
}

template <typename ScalarT>
    requires std::is_integral_v<ScalarT>
void launchAlphabetHistogram(const ScalarT* d_input, ScalarT* d_histo,
                             int32_t nInputs, int32_t divider)
{
    constexpr dim3 blockDim = {1024, 1, 1};
    dim3 gridDim = {uint32_t(ceilDiv(nInputs, blockDim.x)), 1, 1};
    alphabetHistogramKernel<<<gridDim, blockDim>>>(d_input, d_histo, nInputs,
                                                   divider);
    PMPP_DEBUG_CUDA_ERR_CHECK(cudaGetLastError());
}

namespace torch_impl
{
inline auto alphabetHistogram(const torch::Tensor& input, int64_t divider)
    -> torch::Tensor
{
    auto nInputs = input.numel();
    auto histo = torch::zeros({26 / divider}, torch::kInt32);

    switch (input.scalar_type()) {
    case torch::kInt32: {
        pmpp::ops::cuda::launchAlphabetHistogram<int32_t>(
            input.data_ptr<int32_t>(), histo.data_ptr<int32_t>(), nInputs,
            int32_t(divider));
        break;
    }
    default: {
        AT_ERROR("Unsupported dtype: ", input.dtype());
    }
    }

    return histo;
}
}  // namespace torch_impl
}  // namespace pmpp::ops::cuda
