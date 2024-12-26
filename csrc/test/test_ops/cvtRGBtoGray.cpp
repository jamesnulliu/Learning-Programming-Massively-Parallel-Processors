#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>

#include <pmpp/types/cxx_types.hpp>

TEST(OpTest, CvtRGBtoGray)
{
    cudaDeviceReset();
    static auto custom_op = torch::Dispatcher::singleton()
                                .findSchemaOrThrow("pmpp::cvt_rgb_to_gray", "")
                                .typed<torch::Tensor(const torch::Tensor&)>();

    constexpr pmpp::size_t height = 800;
    constexpr pmpp::size_t width = 800;

    torch::Tensor h_image = torch::randint(
        256, {height, width, 3},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    torch::Tensor h_gray = custom_op.call(h_image);

    torch::Tensor d_image =
        h_image.to(torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor d_gray = custom_op.call(d_image);

    EXPECT_TRUE(
        h_gray.equal(d_gray.to(torch::TensorOptions().device(torch::kCPU))));
    
    torch::cuda::synchronize();
    cudaGetLastError();
}


