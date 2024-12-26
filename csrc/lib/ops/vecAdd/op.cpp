#include "pmpp/ops/vec_add.hpp"
#include "pmpp/types/cxx_types.hpp"

namespace pmpp::ops
{

template <>
void launchVecAdd<DeviceType::CPU>(const fp32_t* a, const fp32_t* b, fp32_t* c,
                                   size_t n)
{
#pragma omp for
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

}  // namespace pmpp::ops
