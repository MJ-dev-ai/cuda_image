#include "cuimg/image_io.hpp"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <cstdio>
#include <limits>

namespace cx {

ImageBuffer imread(const std::vector<std::string>& paths) {
    if (paths.empty()) throw std::invalid_argument("imread: Image path list is empty");
    if (paths.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("imread: Too many images");

    std::optional<ImageBuffer> image;
    for (std::size_t n = 0; n < paths.size(); ++n) {
        const auto& image_path = paths[n];
        if (image_path.empty() || image_path.find('\0') != std::string::npos) {
            throw std::invalid_argument("imread: Image path is empty or contains a null character");
        }
        const auto path = std::filesystem::u8path(image_path);
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("imread: Image file does not exist: " + image_path);
        }
        if (!std::filesystem::is_regular_file(path)) {
            throw std::invalid_argument("imread: Path is not a regular file: " + image_path);
        }
    
        int width = 0, height = 0, channels = 0;
        // Release stb memory even if allocation or upload throws.
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
            stbi_load(image_path.c_str(), &width, &height, &channels, STBI_rgb),
            &stbi_image_free);
        if (!pixels) {
            const char* reason = stbi_failure_reason();
            throw std::runtime_error(std::string("imread: Failed to decode image: ") +
                                     (reason ? reason : "Unknown error"));
        }
        if (!image) {
            image.emplace(static_cast<int>(paths.size()), width, height, PixelFormat::RGB);
        } else if (width != image->width() || height != image->height()) {
            throw std::invalid_argument("imread: Batch images must have equal dimensions: " + image_path);
        }
        const auto result = cudaMemcpy(image->data(static_cast<int>(n)), pixels.get(),
                                       image->imageSizeBytes(), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("imread: CUDA upload failed: ") + cudaGetErrorString(result));
        // Keep pageable stb memory alive until the upload finishes.
        const auto completion = cudaStreamSynchronize(nullptr);
        if (completion != cudaSuccess) throw std::runtime_error(cudaGetErrorString(completion));
    }
    return std::move(*image);
}

ImageBuffer imread(const std::string& path) {
    return imread(std::vector<std::string>{path});
}

bool imwrite(const std::vector<std::string>& paths, const ImageBuffer& image, int first) {
    if (first == -1) {
        if (paths.size() != static_cast<std::size_t>(image.batch()))
            throw std::invalid_argument("imwrite: Path count must match image batch size");
        first = 0;
    }
    if (first < 0 || first >= image.batch() || paths.empty() ||
        paths.size() > static_cast<std::size_t>(image.batch() - first))
        throw std::out_of_range("imwrite: Invalid image batch range");
    for (const auto& path : paths) {
        if (path.empty() || path.find('\0') != std::string::npos)
            throw std::invalid_argument("imwrite: Image path is empty or contains a null character");
    }
    if (image.strideBytes() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("imwrite: PNG row stride exceeds int range");

    // Reuse one CPU buffer for the whole batch.
    std::unique_ptr<unsigned char[]> pixels(new unsigned char[image.imageSizeBytes()]);
    for (std::size_t i = 0; i < paths.size(); ++i) {
        const auto* source = image.data(first + static_cast<int>(i));
        if (!source) throw std::invalid_argument("imwrite: Image buffer is empty");
        const auto result = cudaMemcpy(pixels.get(), source, image.imageSizeBytes(),
                                       cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("imwrite: CUDA copy failed: ") + cudaGetErrorString(result));

        if (!stbi_write_png(paths[i].c_str(), image.width(), image.height(), image.channels(),
                            pixels.get(), static_cast<int>(image.strideBytes()))) {
            std::fprintf(stderr, "imwrite: Failed to write PNG image: %s\n", paths[i].c_str());
            return false;
        }
    }
    return true;
}

bool imwrite(const std::string& path, const ImageBuffer& image, int n) {
    // Check here so -1 retains its invalid-index meaning for a single image.
    image.data(n);
    return imwrite(std::vector<std::string>{path}, image, n);
}
} // namespace cx

