#include "cuimg/preprocess.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// Compare batched RGB output against a CPU bilinear reference (half-pixel centers).
void testResize(cx::Size input, cx::Size output, bool preprocess, int channels = 3) {
    const int batch = 3;
    const auto format = channels == 1 ? cx::PixelFormat::GRAY : cx::PixelFormat::RGB;
    cx::ImageBuffer src(batch, input.width, input.height, format);
    cx::ImageBuffer dst(batch, preprocess ? 1 : output.width, preprocess ? 1 : output.height, format);
    std::vector<unsigned char> host(src.sizeBytes());
    std::vector<unsigned char> actual(static_cast<std::size_t>(batch) * output.width * output.height * channels);
    for (int n = 0; n < batch; ++n)
        for (int y = 0; y < input.height; ++y)
            for (int x = 0; x < input.width; ++x)
                for (int c = 0; c < channels; ++c)
                    host[((n * input.height + y) * input.width + x) * channels + c] = (n * 67 + c * 43 + x * 7 + y * 11) % 256;
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream));
    try {
        check(cudaMemcpyAsync(src.data(), host.data(), host.size(), cudaMemcpyHostToDevice, stream));
        if (preprocess) cx::preprocessYolo(src, dst, output, stream);
        else cx::resize(src.data(), dst.data(), batch, input, output, channels, 1, stream);
        if (dst.width() != output.width || dst.height() != output.height) throw std::runtime_error("Wrong dst metadata");
        check(cudaMemcpyAsync(actual.data(), dst.data(), actual.size(), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        for (int n = 0; n < batch; ++n) {
            for (int y = 0; y < output.height; ++y) {
                for (int x = 0; x < output.width; ++x) {
                    const double sx = std::clamp((x + 0.5) * input.width / output.width - 0.5, 0.0, double(input.width - 1));
                    const double sy = std::clamp((y + 0.5) * input.height / output.height - 0.5, 0.0, double(input.height - 1));
                    const int ix = int(sx), iy = int(sy);
                    for (int c = 0; c < channels; ++c) {
                        auto sample = [&](int px, int py) {
                            return host[((n * input.height + std::min(py, input.height - 1)) * input.width
                                + std::min(px, input.width - 1)) * channels + c];
                        };
                        const double fx = sx - ix, fy = sy - iy;
                        const double top = sample(ix, iy) * (1 - fx) + sample(ix + 1, iy) * fx;
                        const double bottom = sample(ix, iy + 1) * (1 - fx) + sample(ix + 1, iy + 1) * fx;
                        const int expected = int(std::floor(top * (1 - fy) + bottom * fy + 0.5));
                        const int value = actual[((n * output.height + y) * output.width + x) * channels + c];
                        const int tolerance = input.width == output.width && input.height == output.height ? 0 : 1;
                        if (std::abs(value - expected) > tolerance) throw std::runtime_error("CPU/GPU pixel mismatch");
                    }
                }
            }
        }
    } catch (...) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); throw; }
    check(cudaStreamDestroy(stream));
}

// Every interpolation must preserve distinct constant channels and batch slices.
void testMethods() {
    cx::ImageBuffer src(3, 9, 7, cx::PixelFormat::RGB), dst(3, 5, 3, cx::PixelFormat::RGB);
    std::vector<unsigned char> host(src.sizeBytes()), actual(dst.sizeBytes());
    for (int n = 0; n < 3; ++n)
        for (int p = 0; p < 63; ++p)
            for (int c = 0; c < 3; ++c) host[(n * 63 + p) * 3 + c] = 20 + n * 60 + c * 17;
    check(cudaMemcpy(src.data(), host.data(), host.size(), cudaMemcpyHostToDevice));
    for (int method = 0; method < 5; ++method) {
        cx::resize(src.data(), dst.data(), 3, {9, 7}, {5, 3}, 3, method, nullptr);
        check(cudaMemcpy(actual.data(), dst.data(), actual.size(), cudaMemcpyDeviceToHost));
        for (int n = 0; n < 3; ++n)
            for (int p = 0; p < 15; ++p)
                for (int c = 0; c < 3; ++c)
                    if (actual[(n * 15 + p) * 3 + c] != 20 + n * 60 + c * 17)
                        throw std::runtime_error("Interpolation changed a constant channel");
    }
    cx::resize(src.data(), src.data(), 3, {9, 7}, {9, 7}, 3, 1, nullptr);
    check(cudaMemcpy(host.data(), src.data(), host.size(), cudaMemcpyDeviceToHost));
    if (host.front() != 20 || host.back() != 174) throw std::runtime_error("In-place no-op changed pixels");
    bool rejected = false;
    try { cx::resize(src.data(), dst.data(), 3, {9, 7}, {5, 3}, 3, 255, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Invalid method accepted");
}

int main() {
    try {
        testResize({7, 5}, {7, 5}, false);
        testResize({7, 5}, {13, 9}, false);
        testResize({17, 11}, {5, 3}, false);
        testResize({1, 1}, {3, 5}, false, 1);
        testResize({13, 9}, {5, 3}, true);
        testResize({5120, 5120}, {2048, 2048}, true);
        testMethods();
        std::cout << "PASS: copy, batch=3, RGB/GRAY, CPU bilinear, borders, preprocess reallocation, 5120->2048, all methods, invalid input\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
