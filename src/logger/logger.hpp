#pragma once

#include <NvInferRuntime.h>

#include <filesystem>
#include <fstream>
#include <mutex>    
#include <string>
#include <string_view>

class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(const std::filesystem::path& log_root);
    ~Logger() override = default;

    // timestamp must be YY-MM-DD:hh-mm-ss; its date selects the log directory.
    void log(std::string_view timestamp, std::string category, nvinfer1::ILogger::Severity severity, const char* msg);
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
    void log(std::string_view timestamp, std::string category, std::string severity_str, const char* msg);

private:
    std::filesystem::path log_root_;
    std::filesystem::path log_dir_;
    std::mutex mutex_;
};

nvinfer1::ILogger::Severity resolveSeverity(const std::string& severity_str);
