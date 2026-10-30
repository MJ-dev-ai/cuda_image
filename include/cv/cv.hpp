#pragma once
#include "cv/image_buffer.hpp"
#include <string>

namespace cv {
inline constexpr int IMREAD_UNCHANGED = -1;
inline constexpr int IMREAD_GRAYSCALE = 0;
inline constexpr int IMREAD_COLOR_BGR = 1;
inline constexpr int IMREAD_ANYDEPTH = 2;
inline constexpr int IMREAD_ANYCOLOR = 4;
inline constexpr int IMREAD_IGNORE_ORIENTATION = 128;
inline constexpr int IMREAD_COLOR_RGB = 256;

// Read into a batch-1 GPU buffer of normalized FP32 CHW pixels. filename is UTF-8.
// Only BGR, RGB and grayscale modes are implemented; failures throw exceptions.
ImageBuffer imread(const std::string& filename, int flags = IMREAD_COLOR_BGR);
bool imwrite(const std::string& filename, const ImageBuffer& image);
} // namespace cv
