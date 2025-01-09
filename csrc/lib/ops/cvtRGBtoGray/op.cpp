#include "pmpp/types/cxx_types.hpp"

namespace pmpp::ops::cpu
{
void launchCvtRGBtoGray(uint8_t* picOut, const uint8_t* picIn, uint32_t nRows,
                        uint32_t nCols)
{
#pragma omp for
    for (uint32_t i = 0; i < nRows * nCols; ++i) {
        uint8_t r = picIn[i * 3];
        uint8_t g = picIn[i * 3 + 1];
        uint8_t b = picIn[i * 3 + 2];
        picOut[i] = uint8_t(0.21F * r + 0.71F * g + 0.07F * b);
    }
}

}  // namespace pmpp::ops::cpu
