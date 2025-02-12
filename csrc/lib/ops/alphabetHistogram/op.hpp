#pragma once
#include "pmpp/pch.hpp"


namespace pmpp::ops::cpu
{
template <typename ScalarT>
    requires std::is_integral_v<ScalarT>
void launchAlphabetHistogram(const ScalarT* input, ScalarT* histo,
                             int32_t nInputs, int32_t divider)
{
    // O(N)
    for (int32_t i = 0; i < nInputs; ++i) {
        auto pos = int32_t(input[i] - 'a');
        if (pos >= 0 && pos < 26) {
            ++histo[pos / divider];
        }
    }
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
        launchAlphabetHistogram<int32_t>(input.data_ptr<int32_t>(),
                                         histo.data_ptr<int32_t>(), nInputs,
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
}  // namespace pmpp::ops::cpu