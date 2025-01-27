#include "../ops.hpp"

namespace pmpp::ops::cpu
{
void launchMatmul(const fp32_t* A, const fp32_t* B, fp32_t* C, size_t m)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            C[i * m + j] = 0;
            for (size_t k = 0; k < m; ++k) {
                C[i * m + j] += A[i * m + k] * B[k * m + j];
            }
        }
    }
}
}  // namespace pmpp::ops::cpu