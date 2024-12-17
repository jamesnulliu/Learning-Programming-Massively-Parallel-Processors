# Minimum required CMake version
cmake_minimum_required(VERSION 3.30)

# Project Name and version
project(VSC-CMAKE-CXX-TEMPLATE VERSION 1.0.0)

# Common utility functions
include(${PROJECT_SOURCE_DIR}/cmake/utils/common.cmake)

log_info(${CMAKE_TOOLCHAIN_FILE})

# @see "./cmake/utils/variables.cmake"
set_default_values(
    # CMake project namespace:
    # All the libs should have a alias with this namespace
    # [NOTE] Change this to your own namespace
    PROJECT_NAMESPACE "pmpp::"
)

# Utility functions for installing libraries:
include(${PROJECT_SOURCE_DIR}/cmake/utils/lib-install.cmake)

# Compiler configurations for cpp, e.g., g++, clang++, msvc(cl), etc.
include(${PROJECT_SOURCE_DIR}/cmake/compilers/cxx-compiler-configs.cmake)

# Compiler configurations for Python and PyTorch(with CUDA):
include(${PROJECT_SOURCE_DIR}/cmake/compilers/cuda-compiler-configs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/libraries/libpython.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/libraries/libtorch.cmake)

# The following libraries are managed by vcpkg
find_package(cxxopts CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(proxy CONFIG REQUIRED)
enable_testing()
find_package(GTest CONFIG REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)