#pragma once
#include "cuimg/image_buffer.hpp"
#include <cstddef>
#include <vector>

namespace cx {

enum class TensorDataType { Float32, Float16 };

// Non-owning view of contiguous RGB NCHW device input (C=3).
// The caller obtains dtype/shape/format from its engine and owns the allocation.
// Packed or vectorized TensorRT formats are outside this interface.
struct TensorView {
    void* data = nullptr;
    std::size_t capacity_bytes = 0;
    int width = 0;
    int height = 0;
    TensorDataType dtype = TensorDataType::Float32;
    int batch = 1;
};

// Retain the actual resize and padding values for mapping detections back.
struct LetterboxTransform {
    int resized_width = 0;
    int resized_height = 0;
    int pad_left = 0;
    int pad_top = 0;
    float scale_x = 1.0F;
    float scale_y = 1.0F;
};

// TODO: Bilinear letterbox (centered, padding=114, allow upscaling),
// color conversion to RGB, /255 normalization and HWC->NCHW in one CUDA kernel.
// Write directly into destination; enqueue inference on the same stream.
// Source and destination allocations must remain valid until stream completion.
// Source/destination batch sizes must match. Returns one transform per image.
std::vector<LetterboxTransform> preprocessYolo(const ImageBuffer& source,
                                 const TensorView& destination,
                                 cudaStream_t stream = nullptr);

#ifdef __CUDACC__
// Device declaration; batch images must be contiguous and have equal shape.
__device__ bool nhwc2nchw(
    const unsigned char* src, unsigned char* dst, 
    int batch, int height, int width, int channels);
#endif

} // namespace cx
