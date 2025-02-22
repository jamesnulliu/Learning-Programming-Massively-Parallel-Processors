# Minimum required CMake version
cmake_minimum_required(VERSION 3.30)

# Project Name and version
project(PMPP VERSION 1.0.0)

# Common utility functions
include(${PROJECT_SOURCE_DIR}/cmake/utils/common.cmake)

log_info("CMake Tookchain File Path: ${CMAKE_TOOLCHAIN_FILE}")

set_default_values(
    PROJECT_NAMESPACE "pmpp::"
    STDOUT_IS_TERMINAL OFF
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
find_package(yaml-cpp CONFIG REQUIRED)
enable_testing()
find_package(GTest CONFIG REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)

set(CMAKE_CTEST_COMMAND ctest)
add_custom_target(check 
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --test-dir ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS all)