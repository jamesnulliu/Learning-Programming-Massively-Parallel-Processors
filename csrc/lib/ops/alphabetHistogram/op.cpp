#include "../ops.hpp"

namespace pmpp::ops::cpu
{
template <>
void launchAlphabetHistogram<int32_t>(const int32_t* input, int32_t* histo,
                                      int32_t nInputs, int32_t divider)
{
    // O(N)
    for (int32_t i = 0; i < nInputs; ++i) {
        int32_t pos = input[i] - 'a';
        if (pos >= 0 && pos < 26) {
            ++histo[pos / divider];
        }
    }
}
}  // namespace pmpp::ops::cpu