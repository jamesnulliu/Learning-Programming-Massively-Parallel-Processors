# Env Variables: CC, CXX, NVCC_CCBIN

set -e  # Exit on error

SOURCE_DIR=.
BUILD_DIR=./build
BUILD_TYPE=Release
CXX_STANDARD=20
CUDA_STANDARD=20
BUILD_SHARED_LIBS=OFF

while [[ $# -gt 0 ]]; do
    case $1 in
        -S|--source-dir)
            SOURCE_DIR=$2; shift ;;
        -B|--build-dir)
            BUILD_DIR=$2; shift ;;
        Release|Debug)
            BUILD_TYPE=$1 ;;
        --stdc++=*)
            CXX_STANDARD="${1#*=}" ;;
        --stdcuda=*)
            CUDA_STANDARD="${1#*=}" ;;
        --prune-env-path)
            # Takes effects only on windows
            source ./csrc/scripts/windows-prune-PATH.sh ;;
        --rm-build-dir)
            rm -rf $BUILD_DIR ;;
        *)
            # @todo Add detailed help message
            echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

cmake -G Ninja -S $SOURCE_DIR -B $BUILD_DIR \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
    -DCMAKE_CUDA_STANDARD=$CUDA_STANDARD 

cmake --build $BUILD_DIR -j $(nproc)