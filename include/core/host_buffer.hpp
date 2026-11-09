#pragma once
#include <cstddef>

class HostBuffer {
private:
    std::size_t capacity_bytes_ = 0;
    void* data_ = nullptr;

public:
    HostBuffer() = default;
    ~HostBuffer() { free(); }

    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    void allocate(std::size_t capacity_bytes);
    void free();

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }

    std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    HostBuffer(HostBuffer&& other) noexcept;
    HostBuffer& operator=(HostBuffer&& other) noexcept;
};