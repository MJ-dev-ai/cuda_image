#pragma once
#include <cstdint>
#include <cstddef>

enum class DType { UInt8, Float32, Float16 };

struct ImagView {
    std::uint8_t* data;
    int width;
    int height;
    int channels;
    std::size_t stride_bytes;
    DType dtype;
};