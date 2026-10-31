#include "cuimg/interpolation.cuh"
#include <cstddef>
#include <math.h>

namespace cx {

namespace {
namespace detail {
__device__ inline float readPixel(const std::uint8_t* src, int width, int height,
                                 int channels, int x, int y, int channel) {
    x = x < 0 ? 0 : (x >= width ? width - 1 : x);
    y = y < 0 ? 0 : (y >= height ? height - 1 : y);
    return src[(static_cast<std::size_t>(y) * width + x) * channels + channel];
}

__device__ inline float cubicWeight(float distance) {
    const float x = fabsf(distance);
    constexpr float a = -0.75F;
    if (x <= 1.0F) return ((a + 2.0F) * x - (a + 3.0F)) * x * x + 1.0F;
    if (x < 2.0F) return ((a * x - 5.0F * a) * x + 8.0F * a) * x - 4.0F * a;
    return 0.0F;
}

__device__ inline float lanczosWeight(float distance) {
    const float x = fabsf(distance);
    if (x < 1.0e-6F) return 1.0F;
    if (x >= 4.0F) return 0.0F;
    constexpr float pi = 3.14159265358979323846F;
    const float t = pi * x;
    return (sinf(t) / t) * (sinf(t * 0.25F) / (t * 0.25F));
}
} // namespace detail
} // namespace

__device__ float interpolateNearest(const std::uint8_t* src, int width, int height,
                                          int channels, float x, float y, int channel) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(height - 1));
    return detail::readPixel(src, width, height, channels,
                             static_cast<int>(floorf(x + 0.5F)),
                             static_cast<int>(floorf(y + 0.5F)), channel);
}

__device__ float interpolateBilinear(const std::uint8_t* src, int width, int height,
                                           int channels, float x, float y, int channel) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(height - 1));
    const int ix = static_cast<int>(floorf(x));
    const int iy = static_cast<int>(floorf(y));
    const float fx = x - ix, fy = y - iy;
    const float top = (1.0F - fx) * detail::readPixel(src, width, height, channels, ix, iy, channel)
                    + fx * detail::readPixel(src, width, height, channels, ix + 1, iy, channel);
    const float bottom = (1.0F - fx) * detail::readPixel(src, width, height, channels, ix, iy + 1, channel)
                       + fx * detail::readPixel(src, width, height, channels, ix + 1, iy + 1, channel);
    return (1.0F - fy) * top + fy * bottom;
}

__device__ float interpolateBicubic(const std::uint8_t* src, int width, int height,
                                          int channels, float x, float y, int channel) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(height - 1));
    const int ix = static_cast<int>(floorf(x)), iy = static_cast<int>(floorf(y));
    float wx[4], wy[4];
    float sum_x = 0.0F, sum_y = 0.0F;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        wx[i] = detail::cubicWeight(x - (ix + i - 1));
        wy[i] = detail::cubicWeight(y - (iy + i - 1));
        sum_x += wx[i];
        sum_y += wy[i];
    }
    float value = 0.0F;
    #pragma unroll
    for (int j = -1; j <= 2; ++j) {
        #pragma unroll
        for (int i = -1; i <= 2; ++i) {
            const float w = wy[j + 1] * wx[i + 1];
            value += w * detail::readPixel(src, width, height, channels, ix + i, iy + j, channel);
        }
    }
    return value / (sum_x * sum_y);
}

// scale_x/y are source pixels per output pixel (src_width/dst_width, etc.).
// Average the source footprint, clipped to the image. Upscaling falls back to bilinear.
__device__ float interpolateArea(const std::uint8_t* src, int width, int height,
                                       int channels, float x, float y, int channel,
                                       float scale_x, float scale_y) {
    if (scale_x < 1.0F || scale_y < 1.0F)
        return interpolateBilinear(src, width, height, channels, x, y, channel);
    const float left = fmaxf(0.0F, x + 0.5F - scale_x * 0.5F);
    const float right = fminf(static_cast<float>(width), x + 0.5F + scale_x * 0.5F);
    const float top = fmaxf(0.0F, y + 0.5F - scale_y * 0.5F);
    const float bottom = fminf(static_cast<float>(height), y + 0.5F + scale_y * 0.5F);
    if (right <= left || bottom <= top)
        return interpolateNearest(src, width, height, channels, x, y, channel);
    const int end_x = static_cast<int>(ceilf(right)), end_y = static_cast<int>(ceilf(bottom));
    float value = 0.0F;
    for (int iy = static_cast<int>(floorf(top)); iy < end_y; ++iy) {
        const float wy = fminf(bottom, static_cast<float>(iy + 1)) - fmaxf(top, static_cast<float>(iy));
        for (int ix = static_cast<int>(floorf(left)); ix < end_x; ++ix) {
            const float wx = fminf(right, static_cast<float>(ix + 1)) - fmaxf(left, static_cast<float>(ix));
            value += wx * wy * detail::readPixel(src, width, height, channels, ix, iy, channel);
        }
    }
    return value / ((right - left) * (bottom - top));
}

__device__ float interpolateLanczos4(const std::uint8_t* src, int width, int height,
                                           int channels, float x, float y, int channel) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(height - 1));
    const int ix = static_cast<int>(floorf(x)), iy = static_cast<int>(floorf(y));
    float wx[8], wy[8];
    float sum_x = 0.0F, sum_y = 0.0F;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        wx[i] = detail::lanczosWeight(x - (ix + i - 3));
        wy[i] = detail::lanczosWeight(y - (iy + i - 3));
        sum_x += wx[i];
        sum_y += wy[i];
    }
    float value = 0.0F;
    #pragma unroll
    for (int j = -3; j <= 4; ++j) {
        #pragma unroll
        for (int i = -3; i <= 4; ++i) {
            const float w = wy[j + 3] * wx[i + 3];
            value += w * detail::readPixel(src, width, height, channels, ix + i, iy + j, channel);
        }
    }
    return value / (sum_x * sum_y);
}

} // namespace cx

