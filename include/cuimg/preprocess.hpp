#pragma once
#include "cuimg/image_buffer.hpp"
#include <cstddef>

namespace cx {

enum class TensorDataType { Float32, Float16 };

// Currently uint8 NHWC bilinear resize only, not full YOLO preprocessing.
// May replace dst allocation: finish prior dst work before calling.
// Keep both buffers alive until stream completion; batches and formats must match.
void preprocessYolo(ImageBuffer& src, ImageBuffer& dst, Size model_input_size, cudaStream_t stream = nullptr);

#ifdef __CUDACC__
// Device declaration; batch images must be contiguous and have equal shape.
__device__ void nhwc2nchw(
    const unsigned char* src, unsigned char* dst, 
    int batch, int height, int width, int channels);
__global__ void resizeKernel(const unsigned char* src, unsigned char* dst,
    int batch, Size src_size, Size dst_size, int channels, std::uint8_t interpolation);
#endif

// Caller owns non-overlapping device buffers and keeps them alive until stream completion.
// dst must hold batch * dst_size.width * dst_size.height * channels bytes.
// Both buffers are tightly packed uint8 NHWC. Equal sizes use a device copy.
void resize(const unsigned char* src, unsigned char* dst, int batch, Size src_size, Size dst_size,
            int channels, std::uint8_t interpolation, cudaStream_t stream);

} // namespace cx
