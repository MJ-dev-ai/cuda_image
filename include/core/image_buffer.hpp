#pragma once

#include <cstddef>
#include <core/device_buffer.hpp>
#include <core/host_buffer.hpp>
#include <cuda_runtime_api.h>

class ImageBuffer {
public:
    ImageBuffer() = default;
    ~ImageBuffer() = default;

    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator=(const ImageBuffer&) = delete;

    ImageBuffer(ImageBuffer&& other) noexcept;
    ImageBuffer& operator=(ImageBuffer&& other) noexcept;

    void allocate(std::size_t size_bytes);
    void free() noexcept;

    void* deviceData(std::size_t offset_bytes = 0);
    const void* deviceData(std::size_t offset_bytes = 0) const;

    void* hostData(std::size_t offset_bytes = 0);
    const void* hostData(std::size_t offset_bytes = 0) const;

    std::size_t sizeBytes() const noexcept { return size_bytes_; }
    std::size_t deviceCapacityBytes() const noexcept {
        return device_buffer_.capacity_bytes();
    }
    std::size_t hostCapacityBytes() const noexcept {
        return host_buffer_.capacity_bytes();
    }

    void upload(cudaStream_t stream);
    void download(cudaStream_t stream);

private:
    DeviceBuffer device_buffer_;
    HostBuffer host_buffer_;
    std::size_t size_bytes_ = 0;
};