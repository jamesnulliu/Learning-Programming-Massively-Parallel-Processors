#include <c10/core/TensorOptions.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <torch/nn/options/distance.h>
#include <torch/torch.h>

#include <pmpp/types/cxx_types.hpp>
#include <pmpp/utils/common.hpp>
#include <torch/types.h>

#include "./op_test.hpp"

using torch::Tensor;
namespace F = torch::nn::functional;

namespace pmpp::test::ops
{
TEST_F(OpTest, CvtRGBtoGray)
{
    static auto custom_op = torch::Dispatcher::singleton()
                                .findSchemaOrThrow("pmpp::cvt_rgb_to_gray", "")
                                .typed<Tensor(const Tensor&)>();

    constexpr pmpp::size_t height = 800;
    constexpr pmpp::size_t width = 800;

    Tensor imgH = torch::randint(
        256, {height, width, 3},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    Tensor grayH = custom_op.call(imgH);

    Tensor imgD = imgH.to(torch::kCUDA);
    Tensor grayD2H = custom_op.call(imgD).to(torch::kCPU);

    Tensor cosSim = F::cosine_similarity(
        grayH.to(torch::kF32).flatten(), grayD2H.to(torch::kF32).flatten(),
        F::CosineSimilarityFuncOptions().dim(0));

    EXPECT_GE(cosSim.item<fp32_t>(), 0.99);
}
}  // namespace pmpp::test::ops
