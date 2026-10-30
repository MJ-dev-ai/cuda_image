#pragma once
#include "cv/cuda_tensor_buffer.hpp"

namespace cv {

class ImageBuffer {
public:
    ImageBuffer(int batch, int channels, int height, int width, int flags);

    std::size_t sizeBytes() const noexcept;
    float* data() noexcept;
    const float* data() const noexcept;

    int batch() const noexcept;
    int channels() const noexcept;
    int height() const noexcept;
    int width() const noexcept;
    // Read mode supplied at creation; pixel storage remains FP32 CHW.
    int flags() const noexcept;

private:
    int batch_size;
    int image_channel;
    int image_height;
    int image_width;
    int image_flags;

    CUDATensorBuffer tensor_buffer;
};

} // namespace cv
