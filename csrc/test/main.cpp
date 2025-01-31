#include <cxxopts.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "./OpTest/OpTest.hpp"

std::shared_ptr<YAML::Node> pmpp::test::ops::OpTest::m_configs = nullptr;

auto main(int argc, char** argv) -> int
{
    testing::InitGoogleTest(&argc, argv);

    cxxopts::Options options("pmpp-test", "Test suite for PMPP");

    options.add_options()("c,config", "Path to the configuration file",
                          cxxopts::value<std::string>()->default_value(
                              "configs/ctests.yml"));
    auto optResult = options.parse(argc, argv);

    pmpp::test::ops::OpTest::setConfigs(optResult["config"].as<std::string>());

    return RUN_ALL_TESTS();
}