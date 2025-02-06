#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>
#include <stdexcept>

#ifdef PMPP_CUDA_ERR_CHECK
    #error "PMPP_CUDA_ERR_CHECK already defined."
#else
    /**
     * @brief Check the given cuda error. Exit with `EXIT_FAILURE` if not
     *        success.
     *        The error message is printed to `stderr`.
     */
    #define PMPP_CUDA_ERR_CHECK(err)                                          \
        do {                                                                  \
            cudaError_t err_ = (err);                                         \
            if (err_ != cudaSuccess) {                                        \
                fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"",     \
                        __FILE__, __LINE__, err, cudaGetErrorString(err_),    \
                        #err);                                                \
                cudaDeviceReset();                                            \
                throw std::runtime_error("CUDA error");                       \
            }                                                                 \
        } while (0)
#endif

#ifdef PMPP_DEBUG_CUDA_ERR_CHECK
    #error "PMPP_DEBUG_CUDA_ERR_CHECK already defined."
#else
    #ifdef NDEBUG
        /**
         * @brief Cuda error check is turned off on Release mode.
         */
        #define PMPP_DEBUG_CUDA_ERR_CHECK(err) ((void) 0)
    #else
        /**
         * @brief Check the given cuda error. Exit with `EXIT_FAILURE` if not
         *        success.
         *        The error message is printed to `stderr`.
         */
        #define PMPP_DEBUG_CUDA_ERR_CHECK(err) PMPP_CUDA_ERR_CHECK(err)
    #endif
#endif

namespace pmpp::cuda
{
template <typename T>
__host__ __device__ void initMemory(T* ptr, size_t n, const T& val)
{
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
}
}  // namespace pmpp::cuda
