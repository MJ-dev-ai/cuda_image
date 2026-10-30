#include "cv/image_buffer.hpp"

#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace cv {

//ImageBuffer
ImageBuffer::ImageBuffer(int batch, int channel, int height, int width, int flags) : tensor_buffer(0) {
    if (batch <= 0 || channel <= 0 || height <= 0 || width <= 0) {
        throw std::invalid_argument("Image shape must be positive integer.");
    }
    std::size_t bytes = sizeof(float);
    for (int dimension : {batch, channel, height, width}) {
        const auto size = static_cast<std::size_t>(dimension);
        if (bytes > std::numeric_limits<std::size_t>::max() / size) {
            throw std::overflow_error("Image buffer size overflow");
        }
        bytes *= size;
    }

    image_channel = channel;
    this->batch_size = batch;
    image_width = width;
    image_height = height;
    image_flags = flags;

    this->tensor_buffer = CUDATensorBuffer(bytes);
}
std::size_t ImageBuffer::sizeBytes() const noexcept { return tensor_buffer.sizeBytes(); }
float* ImageBuffer::data() noexcept { return static_cast<float*>(tensor_buffer.data()); }
const float* ImageBuffer::data() const noexcept { return static_cast<const float*>(tensor_buffer.data()); }

int ImageBuffer::batch() const noexcept { return batch_size; }
int ImageBuffer::channels() const noexcept { return image_channel; }
int ImageBuffer::height() const noexcept { return image_height; }
int ImageBuffer::width() const noexcept { return image_width; }
int ImageBuffer::flags() const noexcept { return image_flags; }

} // namespace cv
