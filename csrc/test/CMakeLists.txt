set(TARGET_NAME "examplePackageTest")

file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_NAME} ${CXX_SOURCES})

target_link_libraries(
    ${TARGET_NAME}
    ${PROJECT_NAMESPACE}examplePackageLib
)