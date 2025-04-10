set(LIB_NAME "PmppTorchOps")

file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# @see "https://cmake.org/cmake/help/latest/module/FindPython.html#commands"
Python_add_library(${LIB_NAME} SHARED ${CXX_SOURCES} ${CUDA_SOURCES})
add_library(${PROJECT_NAMESPACE}${LIB_NAME} ALIAS ${LIB_NAME})

target_include_directories(
    ${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_precompile_headers(
    ${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/pmpp/pch.hpp>
    $<INSTALL_INTERFACE:include/pmpp/pch.hpp>
)

target_link_libraries(
    ${LIB_NAME} 
    PUBLIC 
    Python::Python
    ${TORCH_LIBRARIES}
    fmt::fmt
    msft_proxy
)

target_compile_definitions(${LIB_NAME} 
    PRIVATE $<$<BOOL:${BUILD_SHARED_LIBS}>:PMPP_EXPORT>
    INTERFACE $<$<BOOL:${BUILD_SHARED_LIBS}>:PMPP_IMPORT>
)

# @see "cmake/utils/lib-install.cmake"
install_libraries(${LIB_NAME} ${PROJECT_NAMESPACE} ${LIB_NAME})