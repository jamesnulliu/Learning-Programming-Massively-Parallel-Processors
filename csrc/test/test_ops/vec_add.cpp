#include <gtest/gtest.h>
#include <pmpp/types/cxx_types.hpp>
#include <torch/torch.h>

TEST(OpTest, VecAdd)
{
    static auto custom_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("pmpp::vector_add", "")
            .typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>();

    constexpr pmpp::size_t nElems = 1e8;

    torch::Tensor hA = torch::rand(
        nElems, torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU));
    torch::Tensor hB = torch::rand(
        nElems, torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU));
    torch::Tensor hC1 = custom_op.call(hA, hB);

    torch::Tensor dA = hA.to(torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor dB = hB.to(torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor dC = custom_op.call(dA, dB);
    torch::Tensor hC2 = dC.to(c10::TensorOptions().device(torch::kCPU));

    EXPECT_TRUE(hC1.equal(hC2));
}

TEST(OpTest, VecAdd2)
{
    EXPECT_TRUE(true);
}