#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace pmpp
{
using fp8_t = __nv_fp8_storage_t;
static_assert(sizeof(fp8_t) == 1);
using fp16_t = half;
static_assert(sizeof(fp16_t) == 2);
using bf16_t = nv_bfloat16;
static_assert(sizeof(bf16_t) == 2);
}  // namespace pmpp