#pragma once
#include <cstddef>

namespace cx {

class CUDATensorBuffer {
public:
    CUDATensorBuffer(std::size_t bytes);
    ~CUDATensorBuffer();

    CUDATensorBuffer(const CUDATensorBuffer&) = delete;
    CUDATensorBuffer& operator=(const CUDATensorBuffer&) = delete;

    CUDATensorBuffer(CUDATensorBuffer&& other) noexcept;
    CUDATensorBuffer& operator=(CUDATensorBuffer&& other) noexcept;

    void* data() noexcept;
    const void* data() const noexcept;
    std::size_t sizeBytes() const noexcept;

private:
    void* device_data = nullptr;
    std::size_t size_bytes = 0;
};

} // namespace cx
