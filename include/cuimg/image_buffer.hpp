#pragma once
#include "cuimg/cuda_tensor_buffer.hpp"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace cx {

// Spatial size in width, height order.
struct Size {
    int width = 0;
    int height = 0;

    constexpr bool operator==(const Size& other) const noexcept { return width == other.width && height == other.height; }
    constexpr bool operator!=(const Size& other) const noexcept { return !(*this == other); }
};

enum class PixelFormat { GRAY, RGB, RGBA, GRAY_ALPHA };

// Tightly packed device-resident uint8 NHWC images of equal shape and format.
class ImageBuffer {
public:
    ImageBuffer(Size size, PixelFormat format);
    ImageBuffer(int n, Size size, PixelFormat format);
    ImageBuffer(int width, int height, PixelFormat format);
    ImageBuffer(int n, int width, int height, PixelFormat format);

    std::uint8_t* data() noexcept;
    const std::uint8_t* data() const noexcept;
    std::uint8_t* data(int n);
    const std::uint8_t* data(int n) const;
    int batch() const noexcept;
    Size size() const noexcept;
    std::size_t imageSizeBytes() const noexcept;
    // Total bytes for all images.
    std::size_t sizeBytes() const noexcept;
    std::size_t strideBytes() const noexcept;
    int width() const noexcept;
    int height() const noexcept;
    int channels() const noexcept;
    PixelFormat format() const noexcept;

    // Copy one image at batch index n, accounting for host and device strides.
    // Host memory must remain valid until stream completion; use pinned memory
    // for asynchronous transfer. Downloaded pixels are readable after completion.
    void upload(const std::uint8_t* host_data, std::size_t host_stride_bytes,
                cudaStream_t stream = nullptr, int n = 0);
    void download(std::uint8_t* host_data, std::size_t host_stride_bytes,
                  cudaStream_t stream = nullptr, int n = 0) const;

private:
    int batch_ = 1;
    Size size_;
    int channels_ = 0;
    PixelFormat format_ = PixelFormat::RGB;
    std::size_t stride_bytes_ = 0;
    CUDATensorBuffer buffer_;
};

} // namespace cx



