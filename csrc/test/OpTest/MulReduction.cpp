#include <pmpp/pch.hpp>

#include "./OpTest.hpp"

using torch::Tensor;
namespace f = torch::nn::functional;

namespace pmpp::test::ops
{
TEST_F(OpTest, MulRedection)
{
    const YAML::Node& configs = getConfigs()["OpTest"]["MulReduction"];
    static auto custom_op = torch::Dispatcher::singleton()
                                .findSchemaOrThrow("pmpp::mul_reduction", "")
                                .typed<Tensor(const Tensor&)>();

    for (auto cfg : configs) {
        auto nInputs = cfg["nInputs"].as<pmpp::int64_t>();
        Tensor input = torch::rand({nInputs}).to(torch::kFloat32) * 1.5 + 0.5;

        Tensor resultCPU = custom_op.call(input);
        Tensor resultCUDA = custom_op.call(input.cuda());

        Tensor diff = resultCPU - resultCUDA.cpu();
        EXPECT_LE(diff.abs().max().item<fp32_t>(), 1e-3);
    }
}
}  // namespace pmpp::test::ops