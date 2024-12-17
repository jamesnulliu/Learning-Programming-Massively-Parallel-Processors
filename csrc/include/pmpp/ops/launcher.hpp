#include <string>
#include <map>
#include <functional>
#include <memory>

class Base {
public:
    virtual void doWork() = 0;
    virtual ~Base() = default;

    using Creator = std::function<std::unique_ptr<Base>()>;
    
    static void registerType(const std::string& name, Creator creator);
    static auto create(const std::string& name) -> std::unique_ptr<Base>;

private:
    static std::map<std::string, Creator> registry;
};

// derived.cpp
std::map<std::string, Base::Creator> Base::registry;

void Base::registerType(const std::string& name, Creator creator) {
    registry[name] = creator;
}

auto Base::create(const std::string& name) -> std::unique_ptr<Base> {
    auto it = registry.find(name);
    if (it != registry.end()) {
        return it->second();
    }
    return nullptr;
}

class DerivedA : public Base {
    void doWork() override { /*实现A*/ }
};

// 在程序初始化时注册
static bool registerA = (Base::registerType("A", 
    []() { return std::make_unique<DerivedA>(); }), true);

// user.cpp
void useClass() {
    auto obj = Base::create("A");
    if (obj) {
        obj->doWork();
    }
}