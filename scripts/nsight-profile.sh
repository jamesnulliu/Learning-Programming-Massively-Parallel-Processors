TEST_FILE="./build/test/pmpp_test"
GTEST_FILTER="OpTest.VecAdd"
OUTPUT_FILE="outputs/nsight_profile.ncu-rep"

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test-file)
            TEST_FILE=$2; shift ;;
        -gf|--gtest-filter)
            GTEST_FILTER=$2; shift ;;
        -o|--output-file)
            OUTPUT_FILE=$2; shift ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done


ncu --export $OUTPUT_FILE --force-overwrite \
    --set "full" \
    $TEST_FILE --gtest_filter=$GTEST_FILTER