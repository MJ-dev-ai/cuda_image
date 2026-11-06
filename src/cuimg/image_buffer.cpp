#include "cuimg/image_buffer.hpp"
#include <stdexcept>
#include <limits>

namespace cx {

ImageBuffer::ImageBuffer(int width, int height, PixelFormat format)
    : ImageBuffer(1, Size{width, height}, format) {}

ImageBuffer::ImageBuffer(int n, int width, int height, PixelFormat format)
    : ImageBuffer(n, Size{width, height}, format) {}

ImageBuffer::ImageBuffer(Size size, PixelFormat format)
    : ImageBuffer(1, size, format) {}

ImageBuffer::ImageBuffer(int n, Size size, PixelFormat format)
    : batch_(n), size_(size), format_(format), buffer_(0) {
    const int width = size.width, height = size.height;
    if (n <= 0 || width <= 0 || height <= 0)
        throw std::invalid_argument("Image batch and dimensions must be positive");
    switch (format) {
    case PixelFormat::GRAY: channels_ = 1; break;
    case PixelFormat::GRAY_ALPHA: channels_ = 2; break;
    case PixelFormat::RGB: channels_ = 3; break;
    case PixelFormat::RGBA: channels_ = 4; break;
    default: throw std::invalid_argument("ImageBuffer requires a concrete pixel format");
    }
    const auto max_bytes = std::numeric_limits<std::size_t>::max();
    if (static_cast<std::size_t>(width) > max_bytes / channels_)
        throw std::overflow_error("Image row size overflow");
    stride_bytes_ = static_cast<std::size_t>(width) * channels_;
    if (static_cast<std::size_t>(height) > max_bytes / stride_bytes_)
        throw std::overflow_error("Image buffer size overflow");
    const auto image_bytes = stride_bytes_ * height;
    if (static_cast<std::size_t>(n) > max_bytes / image_bytes)
        throw std::overflow_error("Image batch size overflow");
    buffer_ = CUDATensorBuffer(image_bytes * n);
}
std::uint8_t* ImageBuffer::data() noexcept {
    return static_cast<std::uint8_t*>(buffer_.data());
}
const std::uint8_t* ImageBuffer::data() const noexcept {
    return static_cast<const std::uint8_t*>(buffer_.data());
}
std::uint8_t* ImageBuffer::data(int n) {
    if (n < 0 || n >= batch_) throw std::out_of_range("Image batch index out of range");
    return data() + static_cast<std::size_t>(n) * imageSizeBytes();
}
const std::uint8_t* ImageBuffer::data(int n) const {
    if (n < 0 || n >= batch_) throw std::out_of_range("Image batch index out of range");
    return data() + static_cast<std::size_t>(n) * imageSizeBytes();
}
int ImageBuffer::batch() const noexcept { return batch_; }
Size ImageBuffer::size() const noexcept { return size_; }
std::size_t ImageBuffer::imageSizeBytes() const noexcept { return stride_bytes_ * size_.height; }
std::size_t ImageBuffer::sizeBytes() const noexcept { return buffer_.sizeBytes(); }
std::size_t ImageBuffer::strideBytes() const noexcept { return stride_bytes_; }
int ImageBuffer::width() const noexcept { return size_.width; }
int ImageBuffer::height() const noexcept { return size_.height; }
int ImageBuffer::channels() const noexcept { return channels_; }
PixelFormat ImageBuffer::format() const noexcept { return format_; }

void ImageBuffer::upload(const std::uint8_t* host, std::size_t stride, cudaStream_t stream, int n) {
    auto* destination = data(n);
    if (!host || stride < stride_bytes_) throw std::invalid_argument("Invalid upload buffer or stride");
    const auto result = stride == stride_bytes_
        ? cudaMemcpyAsync(destination, host, imageSizeBytes(), cudaMemcpyHostToDevice, stream)
        : cudaMemcpy2DAsync(destination, stride_bytes_, host, stride, stride_bytes_, size_.height, cudaMemcpyHostToDevice, stream);
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void ImageBuffer::download(std::uint8_t* host, std::size_t stride, cudaStream_t stream, int n) const {
    const auto* source = data(n);
    if (!host || stride < stride_bytes_) throw std::invalid_argument("Invalid download buffer or stride");
    const auto result = stride == stride_bytes_
        ? cudaMemcpyAsync(host, source, imageSizeBytes(), cudaMemcpyDeviceToHost, stream)
        : cudaMemcpy2DAsync(host, stride, source, stride_bytes_, stride_bytes_, size_.height, cudaMemcpyDeviceToHost, stream);
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

} // namespace cx




