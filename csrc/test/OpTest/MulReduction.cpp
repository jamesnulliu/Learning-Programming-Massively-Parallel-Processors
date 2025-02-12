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
        Tensor input = torch::randint(1, 10, {nInputs}).to(torch::kFloat32);

        Tensor resultCPU = custom_op.call(input);
        Tensor resultCUDA = custom_op.call(input.cuda());

        std::cout << resultCPU << std::endl;
        std::cout << resultCUDA << std::endl;

        EXPECT_TRUE(resultCPU.equal(resultCUDA.cpu()));
    }
}
}  // namespace pmpp::test::ops