#include <ATen/core/ATen_fwd.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DeviceType.h>
#include <gtest/gtest.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/optim/optimizer.h>
#include <torch/torch.h>
#include <torch/types.h>

TEST(OpTest, VecAdd)
{
    static auto custom_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("pmpp::vector_add", "")
            .typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>();

    auto hA = torch::rand(
        10000000,
        torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU));
    auto hB = torch::rand(
        10000000,
        torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU));
    auto hC1 = custom_op.call(hA, hB);

    auto dA = hA.to(torch::TensorOptions().device(torch::kCUDA));
    auto dB = hB.to(torch::TensorOptions().device(torch::kCUDA));
    auto dC = custom_op.call(dA, dB);
    auto hC2 = dC.to(c10::TensorOptions().device(torch::kCPU));

    EXPECT_TRUE(hC1.equal(hC2));
}

TEST(OpTest, VecAdd2)
{
    EXPECT_TRUE(true);
}