#pragma once
#include <cstdint>

namespace cv {

inline constexpr std::uint8_t INTER_NEAREST = 0;
inline constexpr std::uint8_t INTER_LINEAR = 1;
inline constexpr std::uint8_t INTER_CUBIC = 2;
inline constexpr std::uint8_t INTER_AREA = 3;
inline constexpr std::uint8_t INTER_LANCZOS4 = 4;

void nearest();
void linear();
void cubic();
void area();
void lanczos4();

} // namespace cv
