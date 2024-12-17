#include <fmt/base.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <pmpp/ops/vec_add.hpp>
#include <pmpp/ops/vec_compare.hpp>

#include <ATen/TensorIterator.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/zero.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <proxy/proxy.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/optim/optimizer.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <pmpp/utils/common.cuh>

auto main(int argc, char** argv) -> int
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}