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