set(TARGET_NAME "pmpp_test")

file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

add_executable(${TARGET_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})

target_link_libraries(
    ${TARGET_NAME} PRIVATE
    -Wl,--no-as-needed ${PROJECT_NAMESPACE}pmpp-torch-ops
    spdlog::spdlog
    GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
    yaml-cpp::yaml-cpp
)

add_test(OpTest ${TARGET_NAME})

file(COPY ${PROJECT_SOURCE_DIR}/../configs DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
