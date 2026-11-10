#include "logger.hpp"

#include <chrono>
#include <cstdio>
#include <format>
#include <initializer_list>
#include <stdexcept>

Logger::Logger(const std::filesystem::path& log_root)
    : log_root_(log_root) {
    const auto now = std::chrono::system_clock::now();
    const auto local_now = std::chrono::zoned_time{std::chrono::current_zone(), now};
    std::string timestamp = std::format("{:%y-%m-%d}", local_now);
    log_dir_ = log_root_ / timestamp;

    std::filesystem::create_directories(log_dir_);
    for (const auto* category : {"app", "trt", "cuda"}) {
        const auto path = log_dir_ / (std::string(category) + ".txt");
        std::ofstream file(path, std::ios::app);

        if (!file) throw std::runtime_error("Failed to create log file: " + path.string());
    }
}

void Logger::log(std::string_view timestamp, std::string category, nvinfer1::ILogger::Severity severity, const char* msg) {
    std::lock_guard lock(mutex_);
    static const std::regex timestamp_pattern(R"(^[0-9]{2}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01]):([01][0-9]|2[0-3])-[0-5][0-9]-[0-5][0-9]$)");
    if (!std::regex_match(timestamp.begin(), timestamp.end(), timestamp_pattern)) {
        throw std::invalid_argument("Invalid timestamp format: " + std::string(timestamp));
    }
    log_dir_ = log_root_ / std::string(timestamp.substr(0, 8));
    std::filesystem::create_directories(log_dir_);
    const auto path = log_dir_ / (category + ".txt");
    std::ofstream file(path, std::ios::app);

    if (!file) throw std::runtime_error("Failed to open log file: " + path.string());

    file << "[" << timestamp << "]\t"
         << category << "\t"
         << static_cast<int>(severity) << "\t"
         << (msg ? msg : "") << "\n";

    file.close();
    if (!file) throw std::runtime_error("Failed to write to log file: " + path.string());
}

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    try {
        const auto now = std::chrono::system_clock::now();
        const auto local_now = std::chrono::zoned_time{std::chrono::current_zone(), now};
        std::string timestamp = std::format("{:%y-%m-%d:%H-%M-%S}", local_now);
        log(timestamp, "trt", severity, msg);
    }
    catch (const std::exception& e) {
        std::fputs("Failed to log message: ", stderr);
        std::fputs(msg ? msg : "", stderr);
        std::fputs("\n", stderr);
    }
}
void Logger::log(std::string_view timestamp, std::string category, std::string severity_str, const char* msg) {
    log(timestamp, category, resolveSeverity(severity_str), msg);
}

nvinfer1::ILogger::Severity resolveSeverity(std::string_view severity)
{
    using Severity = nvinfer1::ILogger::Severity;

    if (severity == "INFO")           return Severity::kINFO;
    if (severity == "WARNING")        return Severity::kWARNING;
    if (severity == "ERROR")          return Severity::kERROR;
    if (severity == "INTERNAL_ERROR") return Severity::kINTERNAL_ERROR;
    if (severity == "VERBOSE")        return Severity::kVERBOSE;

    throw std::invalid_argument("Unknown log severity");
}