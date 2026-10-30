#include "cv/cv.hpp"

#include <cuda_runtime_api.h>
#include <cstddef>
#include <QImageReader>
#include <QImageWriter>
#include <QImage>
#include <QString>
#include <QFileInfo>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cv {
ImageBuffer imread(const std::string& filename, int flags) {
    const QString path = QString::fromUtf8(filename.data(), static_cast<qsizetype>(filename.size()));
    const QFileInfo info(path);
    if (!info.exists()) throw std::runtime_error("Image file does not exist");
    if (!info.isFile()) throw std::runtime_error("Image path is not a file");

    QImageReader reader(path);
    QImage image = reader.read();
    if (image.isNull()) throw std::runtime_error(reader.errorString().toStdString());
    switch (flags) {
        case IMREAD_COLOR_BGR:
            image = image.convertToFormat(QImage::Format_BGR888);
            break;
        case IMREAD_COLOR_RGB:
            image = image.convertToFormat(QImage::Format_RGB888);
            break;
        case IMREAD_GRAYSCALE:
            image = image.convertToFormat(QImage::Format_Grayscale8);
            break;
        default:
            throw std::invalid_argument("Unsupported imread flags");
    }
    if (image.isNull()) throw std::runtime_error("Failed to convert image format");

    const int width = image.width();
    const int height = image.height();
    const int channels = image.format() == QImage::Format_Grayscale8 ? 1 : 3;

    ImageBuffer output(1, channels, height, width, flags);
    std::vector<float> host_data(output.sizeBytes() / sizeof(float));
    const std::size_t plane = static_cast<std::size_t>(width) * height;

    for (int i = 0; i < height; ++i) {
        const uchar* row = image.constScanLine(i);
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < channels; ++k) {
                float value = static_cast<float>(row[static_cast<std::size_t>(j) * channels + k]);
                host_data[plane * k + static_cast<std::size_t>(width) * i + j] = value / 255.0F;
            }
        }
    }
    const cudaError_t result = cudaMemcpy(output.data(), host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    return output;
}

bool imwrite(const std::string& filename, const ImageBuffer& image) {
    if (image.batch() != 1) {
        return false;
    }
    if (!image.data() || image.width() <= 0 || image.height() <= 0) {
        std::fprintf(stderr, "imwrite: Image buffer is empty.\n");
        return false;
    }

    QImage::Format format;
    int channels;
    switch (image.flags()) {
        case IMREAD_COLOR_BGR:
            format = QImage::Format_BGR888;
            channels = 3;
            break;
        case IMREAD_COLOR_RGB:
            format = QImage::Format_RGB888;
            channels = 3;
            break;
        case IMREAD_GRAYSCALE:
            format = QImage::Format_Grayscale8;
            channels = 1;
            break;
        default:
            std::fprintf(stderr, "imwrite: Unsupported image flags. %d\n", image.flags());
            return false;
    }
    if (channels != image.channels()) {
        std::fprintf(stderr, "imwrite: Image channels do not match flags.\n");
        return false;
    }

    const int width = image.width();
    const int height = image.height();
    const std::size_t plane = static_cast<std::size_t>(width) * height;

    std::vector<float> host_data(plane * channels);

    const cudaError_t result = cudaMemcpy(host_data.data(), image.data(), host_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        std::fprintf(stderr, "imwrite: %s\n", cudaGetErrorString(result));
        return false;
    }

    QImage output(width, height, format);
    if (output.isNull()) {
        fprintf(stderr, "imwrite: Failed to allocate output image.\n");
        return false;
    }

    for (int i = 0; i < height; i++) {
        uchar* row = output.scanLine(i);

        for (int j = 0; j < width; j++) {
            for (int c = 0; c < channels; c++) {
                float value = host_data[plane * c + static_cast<std::size_t>(i) * width + j];
                if (!std::isfinite(value)) value = 0.0f;

                row[static_cast<std::size_t>(j) * channels + c] = static_cast<uchar>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
    }
    const QString path = QString::fromUtf8(filename.data(), static_cast<qsizetype>(filename.size()));
    QImageWriter writer(path);
    if (!writer.write(output)) {
        fprintf(stderr, "imwrite: %s: \n", writer.errorString().toUtf8().constData());
        return false;
    }
    return true;
}
} // namespace cv
