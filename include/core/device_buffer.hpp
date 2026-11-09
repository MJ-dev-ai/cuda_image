#pragma once
#include <cstddef>

class DeviceBuffer {
private:
    std::size_t capacity_bytes_ = 0;
    void* data_ = nullptr;

public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { free(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t capacity_bytes);
    void free();

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }

    std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
};