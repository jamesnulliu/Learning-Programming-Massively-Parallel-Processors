set(TARGET_NAME "pmpp_test")

file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

add_executable(${TARGET_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})

include_directories(${GTEST_INCLUDE_DIRS})

target_link_libraries(
    ${TARGET_NAME} PRIVATE
    -Wl,--no-as-needed ${PROJECT_NAMESPACE}pmpp-torch-ops
    msft_proxy
    fmt::fmt
    spdlog::spdlog
    GTest::gtest
    GTest::gtest_main
    GTest::gmock 
    GTest::gmock_main
)

add_test(AllTestsInMain ${TARGET_NAME})