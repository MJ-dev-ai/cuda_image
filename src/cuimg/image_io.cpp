#include "cuimg/image_io.hpp"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <limits>

namespace cx {

void imread(const std::string& image_path, ImageBuffer& image, cudaStream_t stream, int n) {
    if (n < 0 || n >= image.batch()) throw std::out_of_range("imread: Invalid image batch index");
    if (!image.data() || image.format() != PixelFormat::RGB)
        throw std::invalid_argument("imread: Destination must be an allocated RGB buffer");
    if (image_path.empty() || image_path.find('\0') != std::string::npos)
        throw std::invalid_argument("imread: Image path is empty or contains a null character");
    const auto path = std::filesystem::u8path(image_path);
    if (!std::filesystem::exists(path))
        throw std::runtime_error("imread: Image file does not exist: " + image_path);
    if (!std::filesystem::is_regular_file(path))
        throw std::invalid_argument("imread: Path is not a regular file: " + image_path);

    int width = 0, height = 0, channels = 0;
    // Release stb memory after upload completion or on failure.
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load(image_path.c_str(), &width, &height, &channels, STBI_rgb), &stbi_image_free);
    if (!pixels) {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error(std::string("imread: Failed to decode image: ") + (reason ? reason : "Unknown error"));
    }
    if (width != image.width() || height != image.height())
        throw std::invalid_argument("imread: Image dimensions must match destination: " + image_path);
    image.upload(pixels.get(), image.strideBytes(), stream, n);
    // Keep pageable stb memory alive until the upload finishes.
    const auto completion = cudaStreamSynchronize(stream);
    if (completion != cudaSuccess) throw std::runtime_error(cudaGetErrorString(completion));
}

void imread(const std::vector<std::string>& paths, ImageBuffer& image, cudaStream_t stream) {
    if (paths.empty()) throw std::invalid_argument("imread: Image path list is empty");
    if (paths.size() != static_cast<std::size_t>(image.batch()))
        throw std::invalid_argument("imread: Path count must match image batch size");
    for (std::size_t n = 0; n < paths.size(); ++n)
        imread(paths[n], image, stream, static_cast<int>(n));
}

bool imwrite(const std::string& path, const ImageBuffer& image, int n, cudaStream_t stream) {
    if (n < 0 || n >= image.batch()) throw std::out_of_range("imwrite: Invalid image batch index");
    if (path.empty() || path.find('\0') != std::string::npos)
        throw std::invalid_argument("imwrite: Image path is empty or contains a null character");
    if (image.strideBytes() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("imwrite: PNG row stride exceeds int range");
    if (!image.data()) throw std::invalid_argument("imwrite: Image buffer is empty");

    std::unique_ptr<unsigned char[]> pixels(new unsigned char[image.imageSizeBytes()]);
    image.download(pixels.get(), image.strideBytes(), stream, n);
    // PNG encoding requires the download to be complete.
    const auto completion = cudaStreamSynchronize(stream);
    if (completion != cudaSuccess) throw std::runtime_error(cudaGetErrorString(completion));
    if (!stbi_write_png(path.c_str(), image.width(), image.height(), image.channels(),
                        pixels.get(), static_cast<int>(image.strideBytes()))) {
        std::fprintf(stderr, "imwrite: Failed to write PNG image: %s\n", path.c_str());
        return false;
    }
    return true;
}

bool imwrite(const std::vector<std::string>& paths, const ImageBuffer& image, int first, cudaStream_t stream) {
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
    for (std::size_t i = 0; i < paths.size(); ++i) {
        if (!imwrite(paths[i], image, first + static_cast<int>(i), stream)) return false;
    }
    return true;
}
} // namespace cx
