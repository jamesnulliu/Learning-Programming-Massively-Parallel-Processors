#pragma once

#include "pmpp/types/cxx_types.hpp"

namespace pmpp::ops::cpu
{

void launchVecAdd(const fp32_t* a, const fp32_t* b, fp32_t* c, size_t n);

void launchCvtRGBtoGray(uint8_t* picOut, const uint8_t* picIn, uint32_t nRows,
                        uint32_t nCols);

void launchMatmul(const fp32_t* A, const fp32_t* B, fp32_t* C, size_t m);

}  // namespace pmpp::ops::cpu

namespace pmpp::ops::cuda
{

void launchVecAdd(const fp32_t* d_A, const fp32_t* d_B, fp32_t* d_C, size_t n);

void launchCvtRGBtoGray(uint8_t* picOut, const uint8_t* picIn, uint32_t nRows,
                        uint32_t nCols);

void launchMatmul(const fp32_t* dA, const fp32_t* dB, fp32_t* dC, size_t m);

}  // namespace pmpp::ops::cuda