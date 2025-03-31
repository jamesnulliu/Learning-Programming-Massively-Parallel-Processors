#include <pmpp/types/cxx_types.hpp>
#include <torch/torch.h>

#include "./OpTest.hpp"

using torch::Tensor;
namespace f = torch::nn::functional;

namespace pmpp::test::ops
{

TEST_F(OpTest, VecAddv0)
{

    const YAML::Node& configs = getConfigs()["OpTest"]["VecAdd"];

    static auto custom_op = torch::Dispatcher::singleton()
                                .findSchemaOrThrow("pmpp::vector_add_v0", "")
                                .typed<torch::Tensor(const torch::Tensor&,
                                                     const torch::Tensor&)>();

    for (const auto& cfg : configs) {

        auto nElems = cfg["nElems"].as<pmpp::size_t>();

        torch::Tensor matAh = torch::rand(nElems, torch::kF32);
        torch::Tensor matBh = torch::rand(nElems, torch::kF32);
        torch::Tensor matCh = custom_op.call(matAh, matBh);

        ASSERT_TRUE(torch::cuda::is_available());
        torch::Tensor matAd = matAh.to(torch::kCUDA);
        torch::Tensor matBd = matBh.to(matAd.device());
        torch::Tensor matCd2h = custom_op.call(matAd, matBd).to(torch::kCPU);

        Tensor cosSim =
            f::cosine_similarity(matCh.flatten(), matCd2h.flatten(),
                                 f::CosineSimilarityFuncOptions().dim(0));

        EXPECT_TRUE(matCh.allclose(matCd2h));
        EXPECT_GE(cosSim.item<fp32_t>(), 0.99);
    }
}

TEST_F(OpTest, VecAddv1)
{

    const YAML::Node& configs = getConfigs()["OpTest"]["VecAdd"];

    static auto custom_op = torch::Dispatcher::singleton()
                                .findSchemaOrThrow("pmpp::vector_add_v1", "")
                                .typed<torch::Tensor(const torch::Tensor&,
                                                     const torch::Tensor&)>();

    for (const auto& cfg : configs) {

        auto nElems = cfg["nElems"].as<pmpp::size_t>();

        torch::Tensor matAh = torch::rand(nElems, torch::kF32);
        torch::Tensor matBh = torch::rand(nElems, torch::kF32);
        torch::Tensor matCh = custom_op.call(matAh, matBh);

        ASSERT_TRUE(torch::cuda::is_available());
        torch::Tensor matAd = matAh.to(torch::kCUDA);
        torch::Tensor matBd = matBh.to(matAd.device());
        torch::Tensor matCd2h = custom_op.call(matAd, matBd).to(torch::kCPU);

        Tensor cosSim =
            f::cosine_similarity(matCh.flatten(), matCd2h.flatten(),
                                 f::CosineSimilarityFuncOptions().dim(0));

        EXPECT_TRUE(matCh.allclose(matCd2h));
        EXPECT_GE(cosSim.item<fp32_t>(), 0.99);
    }
}

TEST_F(OpTest, VecAddv2)
{

    const YAML::Node& configs = getConfigs()["OpTest"]["VecAdd"];

    static auto custom_op = torch::Dispatcher::singleton()
                                .findSchemaOrThrow("pmpp::vector_add_v2", "")
                                .typed<torch::Tensor(const torch::Tensor&,
                                                     const torch::Tensor&)>();

    for (const auto& cfg : configs) {

        auto nElems = cfg["nElems"].as<pmpp::size_t>();

        torch::Tensor matAh = torch::rand(nElems, torch::kF32);
        torch::Tensor matBh = torch::rand(nElems, torch::kF32);
        torch::Tensor matCh = custom_op.call(matAh, matBh);

        ASSERT_TRUE(torch::cuda::is_available());
        torch::Tensor matAd = matAh.to(torch::kCUDA);
        torch::Tensor matBd = matBh.to(matAd.device());
        torch::Tensor matCd2h = custom_op.call(matAd, matBd).to(torch::kCPU);

        Tensor cosSim =
            f::cosine_similarity(matCh.flatten(), matCd2h.flatten(),
                                 f::CosineSimilarityFuncOptions().dim(0));

        std::cout << std::format("nElems: {}, cosSim: {}\n", nElems,
                                 cosSim.item<fp32_t>());

        // // [NOTE] This won't pass because the kernel is deliberately wrong
        // EXPECT_TRUE(matCh.allclose(matCd2h));
        // EXPECT_GE(cosSim.item<fp32_t>(), 0.99);
    }
}
}  // namespace pmpp::test::ops