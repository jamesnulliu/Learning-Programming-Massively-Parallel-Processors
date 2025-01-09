#pragma once

#include <spdlog/common.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>
#include <utility>

namespace pmpp::basic
{
class LoggableBase
{
public:
    explicit LoggableBase(
        const std::string& loggerName,
        spdlog::level::level_enum level = spdlog::level::info,
        const std::string& pattern =
            "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] [%t] %v")
        : m_logger(spdlog::stdout_color_mt(loggerName))
    {
        m_logger->set_level(level);
        m_logger->set_pattern(pattern);
    }

    auto getLogger() -> std::shared_ptr<spdlog::logger>
    {
        return m_logger;
    }

    void setLogger(std::shared_ptr<spdlog::logger> logger)
    {
        m_logger = std::move(logger);
    }

    void setLevel(spdlog::level::level_enum level)
    {
        m_logger->set_level(level);
    }

    void setPattern(const std::string& pattern)
    {
        m_logger->set_pattern(pattern);
    }

protected:
    std::shared_ptr<spdlog::logger> m_logger;
};
}  // namespace pmpp::basic