#include "../ops.hpp"

namespace pmpp::ops::cpu
{

template <>
void launchConv2D<fp32_t>(const fp32_t* input, const fp32_t* kernel,
                          fp32_t* output, int32_t inputHeight,
                          int32_t inputWidth, int32_t kernelSize)
{
    for (int32_t i = 0; i < inputHeight; ++i) {
        for (int32_t j = 0; j < inputWidth; ++j) {
            fp32_t tmp = 0;
            int32_t startRow = i - kernelSize / 2 < 0 ? 0 : i - kernelSize / 2;
            int32_t startCol = j - kernelSize / 2 < 0 ? 0 : j - kernelSize / 2;
            int32_t endRow = i + kernelSize / 2 >= inputHeight
                                 ? inputHeight - 1
                                 : i + kernelSize / 2;
            int32_t endCol = j + kernelSize / 2 >= inputWidth
                                 ? inputWidth - 1
                                 : j + kernelSize / 2;

            for (int32_t k = startRow; k <= endRow; ++k) {
                for (int32_t l = startCol; l <= endCol; ++l) {
                    tmp += input[k * inputWidth + l] *
                           kernel[(k - i + kernelSize / 2) * kernelSize +
                                  (l - j + kernelSize / 2)];
                }
            }
            output[i * inputWidth + j] = tmp;
        }
    }
}

}  // namespace pmpp::ops::cpu