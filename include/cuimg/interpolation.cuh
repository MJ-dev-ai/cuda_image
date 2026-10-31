#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <math.h>

namespace cx {

inline constexpr std::uint8_t INTER_NEAREST = 0;
inline constexpr std::uint8_t INTER_LINEAR = 1;
inline constexpr std::uint8_t INTER_CUBIC = 2;
inline constexpr std::uint8_t INTER_AREA = 3;
inline constexpr std::uint8_t INTER_LANCZOS4 = 4;

// Sample one channel of a tightly packed uint8 HWC image on the GPU.
// Pass data(n) for a batch slice. Pixel centers have integer coordinates.
// Caller guarantees valid memory, positive dimensions and a valid channel.
// Results are unnormalized floats; cubic/Lanczos may overshoot [0,255].
// Clamp before converting to uint8 if needed.
__device__ float interpolateNearest(const std::uint8_t* src, int width, int height,
                                  int channels, float x, float y, int channel);

__device__ float interpolateBilinear(const std::uint8_t* src, int width, int height,
                                  int channels, float x, float y, int channel);

__device__ float interpolateBicubic(const std::uint8_t* src, int width, int height,
                                  int channels, float x, float y, int channel);

// scale_x/y: source pixels per output pixel. Upscaling uses bilinear.
__device__ float interpolateArea(const std::uint8_t* src, int width, int height,
                                  int channels, float x, float y, int channel,
                                  float scale_x, float scale_y);

__device__ float interpolateLanczos4(const std::uint8_t* src, int width, int height,
                                  int channels, float x, float y, int channel);

} // namespace cx

