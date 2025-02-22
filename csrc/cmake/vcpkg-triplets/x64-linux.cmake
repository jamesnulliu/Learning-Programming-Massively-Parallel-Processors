set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

if(NOT DEFINED ENV{TORCH_VERSION})
    execute_process(
        COMMAND python -c "import torch; print(torch.__version__.split('+')[0])"
        OUTPUT_VARIABLE TORCH_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
    set(TORCH_VERSION $ENV{TORCH_VERSION})
endif()

# Convert version string to comparable number
string(REPLACE "." ";" VERSION_LIST ${TORCH_VERSION})
list(GET VERSION_LIST 0 TORCH_MAJOR)
list(GET VERSION_LIST 1 TORCH_MINOR)
math(EXPR TORCH_VERSION_NUM "${TORCH_MAJOR} * 1000 + ${TORCH_MINOR}")

# Set ABI flag only if torch version < 2.6
if(TORCH_VERSION_NUM LESS 2006)
    set(ENV{CXXFLAGS} "$ENV{CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(ENV{CFLAGS} "$ENV{CFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()