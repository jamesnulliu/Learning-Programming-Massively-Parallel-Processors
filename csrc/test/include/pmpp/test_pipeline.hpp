#pragma once

#include <map>
#include <memory>
#include <proxy/proxy.h>
#include <string>

namespace pmpp
{

PRO_DEF_MEM_DISPATCH(MemTest, test);

struct Testable
    : pro::facade_builder::add_convention<MemTest, bool()>::support_copy<
          pro::constraint_level::nontrivial>::build
{
};

class TestPipeline
{
public:
    explicit TestPipeline()
    {
    }

    void registerTest(const std::string& name, pro::proxy<Testable> test)
    {
        this->registry.emplace(name, test);
    }

private:
    std::map<std::string, pro::proxy<Testable>> registry;
};

}  // namespace pmpp
