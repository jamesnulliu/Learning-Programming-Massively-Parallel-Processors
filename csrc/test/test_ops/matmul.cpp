#include <pmpp/types/cxx_types.hpp>
#include <torch/torch.h>

#include "./op_test.hpp"

using torch::Tensor;
namespace F = torch::nn::functional;

namespace pmpp::test::ops
{

TEST_F(OpTest, Matmul)
{
    static auto custom_op = torch::Dispatcher::singleton()
                                .findSchemaOrThrow("pmpp::matmul", "")
                                .typed<torch::Tensor(const torch::Tensor&,
                                                     const torch::Tensor&)>();

    constexpr pmpp::size_t m = 160;
    constexpr pmpp::size_t n = 160;
    constexpr pmpp::size_t k = 160;

    torch::Tensor matAh = torch::ones({m, n}, torch::kF32);
    torch::Tensor matBh = torch::ones({n, k}, torch::kF32);
    torch::Tensor matCh = custom_op.call(matAh, matBh);

    ASSERT_TRUE(torch::cuda::is_available());
    torch::Tensor matAd = matAh.to(torch::kCUDA);
    torch::Tensor matBd = matBh.to(matAd.device());
    torch::Tensor matCd2h = custom_op.call(matAd, matBd).to(torch::kCPU);

    Tensor cosSim =
        F::cosine_similarity(matCh.flatten(), matCd2h.flatten(),
                             F::CosineSimilarityFuncOptions().dim(0));

    EXPECT_GE(cosSim.item<fp32_t>(), 0.99);
}

}  // namespace pmpp::test::ops