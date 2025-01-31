#include <pmpp/types/cxx_types.hpp>
#include <torch/torch.h>

#include "./OpTest.hpp"

using torch::Tensor;
namespace f = torch::nn::functional;

namespace pmpp::test::ops
{

TEST_F(OpTest, VecAdd)
{
    static auto custom_op = torch::Dispatcher::singleton()
                                .findSchemaOrThrow("pmpp::vector_add", "")
                                .typed<torch::Tensor(const torch::Tensor&,
                                                     const torch::Tensor&)>();

    constexpr pmpp::size_t nElems = 1e3;

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

    EXPECT_GE(cosSim.item<fp32_t>(), 0.99);
}
}  // namespace pmpp::test::ops