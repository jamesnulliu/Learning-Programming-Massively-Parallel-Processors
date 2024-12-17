#pragma once

#ifdef CUDA_ERR_CHECK
    #error "CUDA_ERR_CHECK already defined."
#else
    #ifndef NDEBUG
        #include <cstdio>
        #include <cstdlib>
        #include <cuda_runtime.h>
        #include <cuda_runtime_api.h>
        #include <driver_types.h>

        /**
         * @brief Check the given cuda error. Exit with `EXIT_FAILURE` if not
         *        success.
         *        The error message is printed to `stderr`.
         */
        #define CUDA_ERR_CHECK(err)                                            \
            do {                                                               \
                cudaError_t err_ = (err);                                      \
                if (err_ != cudaSuccess) {                                     \
                    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"",  \
                            __FILE__, __LINE__, err, cudaGetErrorString(err_), \
                            #err);                                             \
                    cudaDeviceReset();                                         \
                    exit(EXIT_FAILURE);                                        \
                }                                                              \
            } while (0)
    #else
        /**
         * @brief Cuda error check is turned off on Release mode.
         */
        #define CUDA_ERR_CHECK(err) ((void) err)
    #endif
#endif
