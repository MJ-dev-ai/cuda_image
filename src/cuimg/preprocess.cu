#include "cuimg/preprocess.hpp"
#include "cuimg/interpolation.cuh"
#include <stdexcept>
#include <limits>

namespace cx {

void preprocessYolo(ImageBuffer& src, ImageBuffer& dst, Size model_input_size, cudaStream_t stream) {
    // Resize-only stage; letterbox and float tensor conversion are not implemented yet.
    if (&src == &dst) throw std::invalid_argument("preprocessYolo requires separate image buffers");
    if (src.batch() != dst.batch() || src.format() != dst.format()) {
        throw std::invalid_argument("Source and destination batch and format must match");
    }
    if (!dst.data() || dst.size() != model_input_size)
        dst = ImageBuffer(src.batch(), model_input_size, src.format());
    resize(src.data(), dst.data(), src.batch(), src.size(), dst.size(), src.channels(), INTER_LINEAR, stream);
    
}

__device__ void nhwc2nchw(const unsigned char* src, unsigned char* dst, int batch, int height, int width, int channels) {
    const std::size_t pixels = static_cast<std::size_t>(height) * width;
    const std::size_t total = static_cast<std::size_t>(batch) * pixels * channels;

    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t i = start; i < total; i += stride) {
        const std::size_t c = i % channels;
        const std::size_t p = (i / channels) % pixels;
        const std::size_t n = i / (channels * pixels);
        dst[(n * channels + c) * pixels + p] = src[i];
    }
}

__global__ void resizeKernel(
    const unsigned char* src, unsigned char* dst, int batch, Size src_size, Size dst_size, int channels, std::uint8_t interpolation) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;

    if (x >= dst_size.width || y >= dst_size.height || n >= batch) {
        return;
    }
    const float src_x = (x + 0.5f) * static_cast<float>(src_size.width) / dst_size.width - 0.5f;
    const float src_y = (y + 0.5f) * static_cast<float>(src_size.height) / dst_size.height - 0.5f;

    const unsigned char* src_n = src + static_cast<std::size_t>(n) * channels * src_size.height * src_size.width;
    for (int i = 0; i < channels; i++) {
        float value = 0.0f;
        switch (interpolation) {
            case INTER_NEAREST:
                value = cx::interpolateNearest(src_n, src_size.width, src_size.height, channels, src_x, src_y, i);
                break;
            case INTER_LINEAR:
                value = cx::interpolateBilinear(src_n, src_size.width, src_size.height, channels, src_x, src_y, i);
                break;
            case INTER_CUBIC:
                value = cx::interpolateBicubic(src_n, src_size.width, src_size.height, channels, src_x, src_y, i);
                break;
            case INTER_AREA:
                value = cx::interpolateArea(src_n, src_size.width, src_size.height, channels, src_x, src_y, i,
                    static_cast<float>(src_size.width) / dst_size.width, static_cast<float>(src_size.height) / dst_size.height);
                break;
            case INTER_LANCZOS4:
                value = cx::interpolateLanczos4(src_n, src_size.width, src_size.height, channels, src_x, src_y, i);
                break;
            default:
                return;
        }
        const auto index = ((static_cast<std::size_t>(n) * dst_size.height + y) * dst_size.width + x) * channels + i;
        dst[index] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, value)) + 0.5f);
    }
}

void resize(const unsigned char* src, unsigned char* dst, int batch, Size src_size, Size dst_size,
            int channels, std::uint8_t interpolation, cudaStream_t stream) {
    if (src_size.width <= 0 || src_size.height <= 0 || dst_size.width <= 0 || dst_size.height <= 0) {
        throw std::invalid_argument("Invalid source or destination size");
    }
    if (!src || !dst || batch <= 0 || channels <= 0) {
        throw std::invalid_argument("Invalid resize buffers, batch or channels");
    }
    if (interpolation > INTER_LANCZOS4) throw std::invalid_argument("Invalid interpolation method");
    if (batch > 65535 || (dst_size.height - 1) / 8 + 1 > 65535)
        throw std::invalid_argument("Resize grid exceeds CUDA dimension limits");
    for (const Size size : {src_size, dst_size}) {
        std::size_t bytes = batch;
        for (const int dimension : {size.width, size.height, channels}) {
            if (bytes > std::numeric_limits<std::size_t>::max() / dimension)
                throw std::overflow_error("Resize buffer size overflow");
            bytes *= dimension;
        }
    }
    if (src_size.width == dst_size.width && src_size.height == dst_size.height) {
        if (src == dst) return;
        const std::size_t bytes = static_cast<std::size_t>(batch) * dst_size.width * dst_size.height * channels;
        const cudaError_t result = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        return;
    }
    if (src == dst) throw std::invalid_argument("Resize requires separate source and destination buffers");
    dim3 block(32, 8, 1);
    dim3 grid((dst_size.width + block.x - 1) / block.x, (dst_size.height + block.y - 1) / block.y, batch);
    resizeKernel<<<grid, block, 0, stream>>>(src, dst, batch, src_size, dst_size, channels, interpolation);
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
} // namespace cx
