#pragma once
#include "cuimg/image_buffer.hpp"
#include <string>
#include <vector>

namespace cx {

// Blocking file I/O, UTF-8 paths. Always decode to tightly packed uint8 RGB.
// Alpha is discarded; grayscale inputs are expanded to RGB.
// Caller supplies an allocated RGB buffer; dimensions must match. No reallocation.
// Single-file reads select slot n (default 0); batch reads call this for each slot.
// Transfers complete on the supplied stream before returning (also for writes).
void imread(const std::string& filename, ImageBuffer& destination, cudaStream_t stream = nullptr, int n = 0);
// Source channel counts may differ. On failure, earlier batch slots may be updated.
void imread(const std::vector<std::string>& filenames, ImageBuffer& destination, cudaStream_t stream = nullptr);
// PNG output. Select one image; n defaults to the first image.
bool imwrite(const std::string& filename, const ImageBuffer& image, int n = 0, cudaStream_t stream = nullptr);
// Calls the single-file writer for each slot; stops on the first failure.
// By default write the whole batch. Explicit first writes a contiguous range.
bool imwrite(const std::vector<std::string>& filenames, const ImageBuffer& image, int first = -1, cudaStream_t stream = nullptr);

} // namespace cx
