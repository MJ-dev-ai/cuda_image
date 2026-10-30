#include "cv/cuda_tensor_buffer.hpp"

#include <cuda_runtime_api.h>
#include <cstdio>
#include <stdexcept>

namespace cv {

// CUDATensorBuffer
CUDATensorBuffer::CUDATensorBuffer(std::size_t bytes) {
    if (bytes == 0) return;

    const cudaError_t result = cudaMalloc(&device_data, bytes);
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));

    size_bytes = bytes;
}
CUDATensorBuffer::~CUDATensorBuffer() {
    if (!device_data) {
        size_bytes = 0;
        return;
    }
    const cudaError_t result = cudaFree(device_data);
    if (result != cudaSuccess) std::fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(result));
}

CUDATensorBuffer::CUDATensorBuffer(CUDATensorBuffer&& other) noexcept {
    device_data = other.device_data;
    size_bytes = other.size_bytes;

    other.device_data = nullptr;
    other.size_bytes = 0;
}

CUDATensorBuffer& CUDATensorBuffer::operator=(CUDATensorBuffer&& other) noexcept {
    if (this == &other) return *this;
    if (device_data) {
        const cudaError_t result = cudaFree(device_data);
        if (result != cudaSuccess) {
            std::fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(result));
        }
    }
    device_data = other.device_data;
    size_bytes = other.size_bytes;

    other.device_data = nullptr;
    other.size_bytes = 0;

    return *this;
}

void* CUDATensorBuffer::data() noexcept { return device_data; }
const void* CUDATensorBuffer::data() const noexcept { return device_data; }
std::size_t CUDATensorBuffer::sizeBytes() const noexcept { return size_bytes; }

} // namespace cv
