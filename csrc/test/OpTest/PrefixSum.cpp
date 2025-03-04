#include <pmpp/types/cxx_types.hpp>
#include <torch/torch.h>

#include "./OpTest.hpp"

using torch::Tensor;
namespace f = torch::nn::functional;

namespace pmpp::test::ops
{
TEST_F(OpTest, PrefixSum)
{
    const YAML::Node& configs = getConfigs()["OpTest"]["PrefixSum"];

    static auto custom_op = torch::Dispatcher::singleton()
                                .findSchemaOrThrow("pmpp::prefix_sum", "")
                                .typed<Tensor(const Tensor&)>();

    for (auto cfg : configs) {
        auto nInputs = cfg["nInputs"].as<pmpp::int64_t>();

        spdlog::info("nInputs: {}", nInputs);

        Tensor input = torch::rand({nInputs}, torch::kFloat32);
        Tensor outputCPU = custom_op.call(input);
        Tensor outputCUDA = custom_op.call(input.cuda());

        Tensor cosSim = f::cosine_similarity(
            outputCPU.flatten(), outputCUDA.cpu().flatten(),
            f::CosineSimilarityFuncOptions().dim(0));

        EXPECT_GE(cosSim.item<fp32_t>(), 0.99);
    }
}
}  // namespace pmpp::test::ops