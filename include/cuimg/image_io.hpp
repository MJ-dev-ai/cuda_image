#pragma once
#include "cuimg/image_buffer.hpp"
#include <string>
#include <vector>

namespace cx {

// Blocking file I/O, UTF-8 paths. Always decode to tightly packed uint8 RGB.
// Alpha is discarded; grayscale inputs are expanded to RGB.
ImageBuffer imread(const std::string& filename);
// All files must have the same dimensions; source channel counts may differ.
ImageBuffer imread(const std::vector<std::string>& filenames);
// PNG output. Select one image; n defaults to the first image.
bool imwrite(const std::string& filename, const ImageBuffer& image, int n = 0);
// By default write the whole batch. Explicit first writes a contiguous range.
bool imwrite(const std::vector<std::string>& filenames, const ImageBuffer& image, int first = -1);

} // namespace cx
