#include <pmpp/types/cxx_types.hpp>
#include <torch/torch.h>

#include "./OpTest.hpp"

using torch::Tensor;
namespace f = torch::nn::functional;

namespace pmpp::test::ops
{
TEST_F(OpTest, AlphabetHistogram)
{
    const YAML::Node& configs = getConfigs()["OpTest"]["AlphabetHistogram"];

    static auto custom_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("pmpp::alphabet_histogram", "")
            .typed<Tensor(const Tensor&, int64_t)>();

    for (auto cfg : configs) {
        auto nInputs = cfg["nInputs"].as<pmpp::int64_t>();
        auto divider = cfg["divider"].as<int64_t>();

        Tensor input = torch::randint(0, 26, {nInputs}, torch::kInt32);

        Tensor histoCPU = custom_op.call(input, divider);

        Tensor histoCUDA = custom_op.call(input.cuda(), divider);

        EXPECT_TRUE(histoCPU.equal(histoCUDA.cpu()));
    }
}
}  // namespace pmpp::test::ops