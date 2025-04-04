# Env Variables: CC, CXX, NVCC_CCBIN, VCPKG_HOME

set -e  # Exit on error

export CC=gcc
export CXX=g++
export TORCH_VERSION=2.6.0

SOURCE_DIR=./csrc
BUILD_DIR=./build
BUILD_TYPE=Release
CXX_STANDARD=20
CUDA_STANDARD=20
BUILD_SHARED_LIBS=OFF
VCPKG_HOME=$VCPKG_HOME

if [ -t 1 ]; then
    STDOUT_IS_TERMINAL=ON
    export GTEST_COLOR=yes
else
    STDOUT_IS_TERMINAL=OFF
    export GTEST_COLOR=no
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        -S|--source-dir)
            SOURCE_DIR=$2; shift ;;
        -B|--build-dir)
            BUILD_DIR=$2; shift ;;
        Release|Debug|RelWithDebInfo|RD)
            BUILD_TYPE=${1/RD/RelWithDebInfo} ;;
        --stdc++=*)
            CXX_STANDARD="${1#*=}" ;;
        --stdcuda=*)
            CUDA_STANDARD="${1#*=}" ;;
        --prune-env-path)
            # Takes effects only on windows
            source ./scripts/windows-prune-PATH.sh ;;
        --rm-build-dir)
            rm -rf $BUILD_DIR ;;
        --vcpkg-home|--vcpkg-dir|--vcpkg-root)
            VCPKG_HOME=$2; shift ;;
        *)
            # @todo Add detailed help message
            echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

cmake -S $SOURCE_DIR -B $BUILD_DIR -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_HOME/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_OVERLAY_TRIPLETS="csrc/cmake/vcpkg-triplets" \
    -DVCPKG_TARGET_TRIPLET="x64-linux" \
    -DSTDOUT_IS_TERMINAL=$STDOUT_IS_TERMINAL \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
    -DCMAKE_CUDA_STANDARD=$CUDA_STANDARD 

cmake --build $BUILD_DIR --parallel 12 --target all check
