#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <pmpp/utils/common.cuh>

namespace pmpp::test::ops
{
class OpTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        torch::cuda::synchronize();
        PMPP_CUDA_ERR_CHECK(cudaGetLastError());
    }

    void TearDown() override
    {
        torch::cuda::synchronize();
        PMPP_CUDA_ERR_CHECK(cudaGetLastError());
    }

    static auto logger() -> std::shared_ptr<spdlog::logger>
    {
        static auto logger = spdlog::stdout_color_mt("OpTest");
        return logger;
    }
};

}  // namespace pmpp::test::ops