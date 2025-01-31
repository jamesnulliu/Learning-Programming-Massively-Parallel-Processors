#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

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

public:
    static void setConfigs(const std::string& filePath)
    {
        m_configs = std::make_shared<YAML::Node>(YAML::LoadFile(filePath));
    }

    static void setConfigs(const YAML::Node& configs)
    {
        m_configs = std::make_shared<YAML::Node>(configs);
    }

    [[nodiscard]] static auto getConfigs() -> const YAML::Node&
    {
        return *m_configs;
    }

private:
    static std::shared_ptr<YAML::Node> m_configs;
};

}  // namespace pmpp::test::ops