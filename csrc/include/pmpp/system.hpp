#pragma once

#if defined(_WIN32) || defined(__WIN32__)
    #if defined(PMPP_EXPORT)
        #define PMPP_API __declspec(dllexport)
    #elif defined(PMPP_IMPORT)
        #define PMPP_API __declspec(dllimport)
    #else
        #define PMPP_API
    #endif
#else
    #if defined(PMPP_EXPORT)
        #define PMPP_API __attribute__((visibility("default")))
    #else
        #define PMPP_API
    #endif
#endif

#if defined(_WIN32) || defined(__WIN32__)
    #define PMPP_CALL __cdecl
#else
    #define PMPP_CALL
#endif

#ifdef __cplusplus
    #define PMPP_EXTERN_C extern "C"
    #define PMPP_EXTERN_C_BEGIN                                               \
        extern "C"                                                            \
        {
    #define PMPP_EXTERN_C_END }
#else
    #define PMPP_EXTERN_C
    #define PMPP_EXTERN_C_BEGIN
    #define PMPP_EXTERN_C_END
#endif