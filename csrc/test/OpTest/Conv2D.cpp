#include <pmpp/types/cxx_types.hpp>
#include <torch/torch.h>

#include "./OpTest.hpp"

using torch::Tensor;
namespace f = torch::nn::functional;

namespace pmpp::test::ops
{
TEST_F(OpTest, Conv2D)
{
    const YAML::Node& configs = getConfigs()["OpTest"]["Conv2D"];

    static auto custom_op = torch::Dispatcher::singleton()
                                .findSchemaOrThrow("pmpp::conv2d", "")
                                .typed<Tensor(const Tensor&, const Tensor&)>();

    for (auto testConfig : configs) {
        auto inputHeight = testConfig["inputHeight"].as<pmpp::int64_t>();
        auto inputWidth = testConfig["inputWidth"].as<pmpp::int64_t>();
        auto kernelSize = testConfig["kernelSize"].as<pmpp::int64_t>();

        Tensor h_input = torch::randn(
            {inputHeight, inputWidth},
            torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU));

        Tensor h_kernel = torch::randn(
            {kernelSize, kernelSize},
            torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU));

        Tensor h_output = custom_op.call(h_input, h_kernel);

        Tensor d_input = h_input.to(torch::kCUDA);
        Tensor d_kernel = h_kernel.to(torch::kCUDA);
        Tensor d_output = custom_op.call(d_input, d_kernel);

        Tensor cosSim =
            f::cosine_similarity(h_output.flatten(), d_output.cpu().flatten(),
                                 f::CosineSimilarityFuncOptions().dim(0));

        EXPECT_GE(cosSim.item<fp32_t>(), 0.99);

        torch::cuda::synchronize();
    }
}
}  // namespace pmpp::test::ops