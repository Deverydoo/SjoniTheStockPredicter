#pragma once

// ============================================================================
// ArgusTrader Logging Utilities
// ============================================================================

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <memory>
#include <string>
#include <filesystem>

namespace argus {
namespace logging {

inline void init(const std::string& log_dir = "./logs",
                 const std::string& level = "INFO") {
    // Create log directory if it doesn't exist
    std::filesystem::create_directories(log_dir);

    // Console sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);

    // File sink
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_dir + "/argus.log",
        100 * 1024 * 1024,  // 100 MB
        10                   // 10 files
    );
    file_sink->set_level(spdlog::level::debug);

    // Create logger
    auto logger = std::make_shared<spdlog::logger>(
        "argus",
        spdlog::sinks_init_list{console_sink, file_sink}
    );

    // Set level
    if (level == "TRACE" || level == "trace") {
        logger->set_level(spdlog::level::trace);
    } else if (level == "DEBUG" || level == "debug") {
        logger->set_level(spdlog::level::debug);
    } else if (level == "WARN" || level == "warn") {
        logger->set_level(spdlog::level::warn);
    } else if (level == "ERROR" || level == "error") {
        logger->set_level(spdlog::level::err);
    } else {
        logger->set_level(spdlog::level::info);
    }

    // Set pattern: [timestamp] [level] [logger] message
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    // Set as default
    spdlog::set_default_logger(logger);

    spdlog::info("Logging initialized - level: {}", level);
}

inline void shutdown() {
    spdlog::shutdown();
}

}  // namespace logging
}  // namespace argus
