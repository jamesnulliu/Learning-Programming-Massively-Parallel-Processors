#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <pmpp/types/cxx_types.hpp>
#include <torch/cuda.h>
#include <torch/torch.h>

TEST(OpTest, VecAdd)
{
    cudaDeviceReset();
    static auto custom_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("pmpp::vector_add", "")
            .typed<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>();

    constexpr pmpp::size_t nElems = 1e3;

    torch::Tensor hA = torch::rand(
        nElems, torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU));
    torch::Tensor hB = torch::rand(
        nElems, torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU));
    torch::Tensor hC1 = custom_op.call(hA, hB);

    ASSERT_TRUE(torch::cuda::is_available());

    auto device = torch::Device(torch::kCUDA, 0);

    torch::Tensor dA = hA.to(device);
    torch::Tensor dB = hB.to(device);
    torch::Tensor dC = custom_op.call(dA, dB);
    torch::Tensor hC2 = dC.to(c10::TensorOptions().device(torch::kCPU));

    EXPECT_TRUE(hC1.equal(hC2));

    torch::cuda::synchronize();
    cudaGetLastError();
}
