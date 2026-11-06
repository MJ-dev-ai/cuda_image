#include "cuimg/preprocess.hpp"
#include <QApplication>
#include <QCloseEvent>
#include <QComboBox>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QImage>
#include <QFileInfo>
#include <QFile>
#include <QTextStream>
#include <memory>
#include <cstring>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>
#include <QThread>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <vector>

namespace {
struct Result {
    std::vector<QImage> sources;
    std::vector<QImage> images;
    QString error;
    double elapsed_ms = 0, prepare_ms = 0;
    float upload_ms = 0, resize_ms = 0, download_ms = 0;
    bool reused_input = false;
};

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// Reusable pinned staging storage; only grows when a larger batch arrives.
struct PinnedBuffer {
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    unsigned char* data = nullptr;
    std::size_t capacity = 0;
    ~PinnedBuffer() { if (data) cudaFreeHost(data); }
    void reserve(std::size_t bytes) {
        if (bytes <= capacity) return;
        void* replacement = nullptr;
        checkCuda(cudaMallocHost(&replacement, bytes));
        if (data) cudaFreeHost(data);
        data = static_cast<unsigned char*>(replacement);
        capacity = bytes;
    }
};

// Persistent resources belong exclusively to the worker thread.
class BatchWorker : public QObject {
public:
    ~BatchWorker() override {
        if (stream) cudaStreamSynchronize(stream);
        for (auto event : events) if (event) cudaEventDestroy(event);
        if (stream) cudaStreamDestroy(stream);
    }

    // One kernel launch processes all images. File batches require equal source dimensions.
    Result process(const QStringList& paths, int revision, int width, int height, int batch, int method, bool preprocess) {
        Result result;
        QElapsedTimer timer;
        timer.start();
        try {
            if (!stream) checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            for (auto& event : events) if (!event) checkCuda(cudaEventCreate(&event));
            const bool reload = revision != loaded_revision || batch != loaded_batch;
            if (reload) {
                input_ready = false;
                std::vector<QImage> decoded;
                for (int n = 0; n < batch; ++n) {
                    QImage image;
                    if (paths.isEmpty()) {
                        image = QImage(5120, 5120, QImage::Format_RGB888);
                        if (image.isNull()) throw std::runtime_error("Cannot allocate pattern");
                        for (int y = 0; y < image.height(); ++y) {
                            auto* row = image.scanLine(y);
                            for (int x = 0; x < image.width(); ++x) {
                                row[x * 3] = (x * 255 / 5119 + n * 37) % 256;
                                row[x * 3 + 1] = (y * 255 / 5119 + n * 61) % 256;
                                row[x * 3 + 2] = ((x / 160 + y / 160 + n) % 2) ? 230 : 25;
                            }
                        }
                    } else {
                        image = QImage(paths.at(n)).convertToFormat(QImage::Format_RGB888);
                        if (image.isNull()) throw std::runtime_error(("Cannot decode: " + paths.at(n)).toStdString());
                    }
                    if (!decoded.empty() && image.size() != decoded.front().size())
                        throw std::invalid_argument("Batch images must have equal width and height. Select matching images.");
                    decoded.push_back(std::move(image));
                }
                const int sw = decoded.front().width(), sh = decoded.front().height();
                if (!src || src->batch() != batch || src->width() != sw || src->height() != sh)
                    src = std::make_unique<cx::ImageBuffer>(batch, sw, sh, cx::PixelFormat::RGB);
                upload.reserve(src->sizeBytes());
                for (int n = 0; n < batch; ++n)
                    for (int y = 0; y < sh; ++y)
                        std::memcpy(upload.data + n * src->imageSizeBytes() + y * src->strideBytes(),
                                    decoded[n].constScanLine(y), src->strideBytes());
                sources = std::move(decoded);
                loaded_revision = revision;
                loaded_batch = batch;
            }
            if (!src) throw std::runtime_error("No input batch");
            if (!dst || dst->batch() != batch || dst->width() != width || dst->height() != height)
                dst = std::make_unique<cx::ImageBuffer>(batch, width, height, cx::PixelFormat::RGB);
            download.reserve(dst->sizeBytes());
            result.sources = sources;
            result.images.resize(batch);
            for (auto& image : result.images) {
                image = QImage(width, height, QImage::Format_RGB888);
                if (image.isNull()) throw std::runtime_error("Cannot allocate output image");
            }
            result.prepare_ms = timer.nsecsElapsed() / 1e6;
            result.reused_input = input_ready;
            checkCuda(cudaEventRecord(events[0], stream));
            if (!input_ready)
                checkCuda(cudaMemcpyAsync(src->data(), upload.data, src->sizeBytes(), cudaMemcpyHostToDevice, stream));
            checkCuda(cudaEventRecord(events[1], stream));
            if (preprocess) cx::preprocessYolo(*src, *dst, {width, height}, stream);
            else cx::resize(src->data(), dst->data(), batch, {src->width(), src->height()}, {width, height}, 3,
                            static_cast<std::uint8_t>(method), stream);
            checkCuda(cudaEventRecord(events[2], stream));
            checkCuda(cudaMemcpyAsync(download.data, dst->data(), dst->sizeBytes(), cudaMemcpyDeviceToHost, stream));
            checkCuda(cudaEventRecord(events[3], stream));
            checkCuda(cudaStreamSynchronize(stream));
            input_ready = true;
            if (!result.reused_input) checkCuda(cudaEventElapsedTime(&result.upload_ms, events[0], events[1]));
            checkCuda(cudaEventElapsedTime(&result.resize_ms, events[1], events[2]));
            checkCuda(cudaEventElapsedTime(&result.download_ms, events[2], events[3]));
            for (int n = 0; n < batch; ++n)
                for (int y = 0; y < height; ++y)
                    std::memcpy(result.images[n].scanLine(y), download.data + n * dst->imageSizeBytes() + y * dst->strideBytes(),
                                dst->strideBytes());
            result.elapsed_ms = timer.nsecsElapsed() / 1e6;
        } catch (const std::exception& error) {
            if (stream) cudaStreamSynchronize(stream);
            input_ready = false;
            loaded_revision = -1;
            result.error = QString::fromUtf8(error.what());
        } catch (...) {
            if (stream) cudaStreamSynchronize(stream);
            input_ready = false;
            loaded_revision = -1;
            result.error = "Unknown processing error";
        }
        return result;
    }
private:
    cudaStream_t stream = nullptr;
    cudaEvent_t events[4]{};
    std::unique_ptr<cx::ImageBuffer> src, dst;
    PinnedBuffer upload, download;
    std::vector<QImage> sources;
    int loaded_revision = -1, loaded_batch = 0;
    bool input_ready = false;
};

class ResizeWindow : public QWidget {
public:
    ResizeWindow() {
        worker = new BatchWorker;
        worker->moveToThread(&worker_thread);
        connect(&worker_thread, &QThread::finished, worker, &QObject::deleteLater);
        worker_thread.start();
        setWindowTitle("CUDA resize test");
        resize(1100, 720);
        auto* layout = new QVBoxLayout(this);
        auto* controls = new QHBoxLayout;
        open = new QPushButton("Open batch images");
        pattern = new QPushButton("5120 x 5120 patterns");
        run_button = new QPushButton("Run on GPU");
        save_button = new QPushButton("Save result");
        save_button->setEnabled(false);
        controls->addWidget(open);
        controls->addWidget(pattern);
        controls->addWidget(run_button);
        controls->addWidget(save_button);
        layout->addLayout(controls);
        auto* options = new QHBoxLayout;
        width_box = new QSpinBox;
        height_box = new QSpinBox;
        batch_box = new QSpinBox;
        for (auto* box : {width_box, height_box}) { box->setRange(1, 8192); box->setValue(2048); }
        batch_box->setRange(1, 64);
        method_box = new QComboBox;
        method_box->addItems({"Nearest", "Linear", "Cubic", "Area", "Lanczos4"});
        method_box->setCurrentIndex(1);
        mode_box = new QComboBox;
        mode_box->addItems({"resize", "preprocessYolo (resize only)"});
        options->addWidget(new QLabel("Width")); options->addWidget(width_box);
        options->addWidget(new QLabel("Height")); options->addWidget(height_box);
        options->addWidget(new QLabel("Batch")); options->addWidget(batch_box);
        options->addWidget(method_box); options->addWidget(mode_box);
        layout->addLayout(options);
        path_label = new QLabel("Source: generated RGB pattern (5120 x 5120)");
        path_label->setWordWrap(true);
        layout->addWidget(path_label);
        auto* captions = new QHBoxLayout;
        captions->addWidget(new QLabel("Source (fit to view)"));
        captions->addWidget(new QLabel("GPU output (fit to view)"));
        layout->addLayout(captions);
        auto* views = new QHBoxLayout;
        source_view = new QLabel("Source");
        output_view = new QLabel("GPU result");
        for (auto* view : {source_view, output_view}) {
            view->setAlignment(Qt::AlignCenter);
            view->setMinimumSize(200, 200);
            view->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
            views->addWidget(view);
        }
        layout->addLayout(views, 1);
        slice_box = new QComboBox;
        layout->addWidget(slice_box);
        status = new QLabel("Select equally sized images for a real batch, or generate distinct patterns. Reselect files to reload changed content.");
        status->setWordWrap(true);
        status->setMinimumHeight(status->fontMetrics().lineSpacing() * 3);
        layout->addWidget(status);
        connect(open, &QPushButton::clicked, this, [this] {
            const QStringList selected = QFileDialog::getOpenFileNames(this, "Open batch images", {}, "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)");
            if (selected.isEmpty()) return;
            if (selected.size() > 64) { status->setText("Select up to 64 images per batch."); return; }
            paths = selected;
            ++revision;
            batch_box->setValue(paths.size());
            batch_box->setEnabled(false);
            path_label->setText(QString("Selected %1 images. First: %2").arg(paths.size()).arg(paths.front()));
            status->setText("Batch selected. Click Run on GPU to upload and process.");
        });
        connect(pattern, &QPushButton::clicked, this, [this] {
            paths.clear(); ++revision; batch_box->setEnabled(true);
            path_label->setText("Source: distinct generated RGB patterns (5120 x 5120)");
        });
        connect(mode_box, &QComboBox::currentIndexChanged, this, [this](int index) {
            method_box->setEnabled(index == 0);
        });
        connect(run_button, &QPushButton::clicked, this, &ResizeWindow::start);
        connect(slice_box, &QComboBox::currentIndexChanged, this, [this] { refresh(); });
        connect(save_button, &QPushButton::clicked, this, [this] {
            const QString destination = QFileDialog::getSaveFileName(this, "Save result", {}, "PNG (*.png)");
            if (!destination.isEmpty() && !result.images.at(slice_box->currentIndex()).save(destination, "PNG"))
                status->setText("Failed to save result");
        });
    }

    ~ResizeWindow() override { worker_thread.quit(); worker_thread.wait(); }

    void finish(Result completed) {
            running = false;
            result = std::move(completed);
            run_button->setEnabled(true);
            open->setEnabled(true); pattern->setEnabled(true);
            slice_box->clear();
            if (!result.error.isEmpty()) {
                source_view->clear(); output_view->clear();
                status->setText("Error: " + result.error);
            } else {
                for (int n = 0; n < static_cast<int>(result.images.size()); ++n)
                    slice_box->addItem(paths.isEmpty() ? QString("Pattern %1").arg(n) : QString("%1: %2").arg(n).arg(QFileInfo(paths[n]).fileName()));
                save_button->setEnabled(true);
                const auto& output = result.images.front();
                status->setText(QString("Batch %1 | %2 x %3 -> %4 x %5 | Total %6 ms (%7 ms/image)\n"
                                        "Prepare %8 ms | Upload %9 ms%10 | Resize %11 ms | Download %12 ms | includes CPU preview copies, excludes rendering")
                    .arg(result.images.size()).arg(result.sources.front().width()).arg(result.sources.front().height())
                    .arg(output.width()).arg(output.height()).arg(result.elapsed_ms, 0, 'f', 2)
                    .arg(result.elapsed_ms / result.images.size(), 0, 'f', 2).arg(result.prepare_ms, 0, 'f', 2)
                    .arg(result.upload_ms, 0, 'f', 2).arg(result.reused_input ? " (cached)" : "")
                    .arg(result.resize_ms, 0, 'f', 2).arg(result.download_ms, 0, 'f', 2));
                refresh();
            }
            if (smoke) {
                QFile timing_file(smoke_dir + "/batch-timing.txt");
                if (timing_file.open(QIODevice::WriteOnly | QIODevice::Append)) {
                    QTextStream log(&timing_file);
                    log << "run=" << smoke_run << " batch=" << result.images.size() << " total_ms=" << result.elapsed_ms
                        << " prepare_ms=" << result.prepare_ms << " upload_ms=" << result.upload_ms
                        << " resize_ms=" << result.resize_ms << " download_ms=" << result.download_ms << '\n';
                }
                if (smoke_run++ == 0 && result.error.isEmpty()) {
                    QTimer::singleShot(0, this, &ResizeWindow::start);
                    return;
                }
                const bool ok = result.error.isEmpty() && result.images.front().save(smoke_dir + "/gui-result.png");
                grab().save(smoke_dir + "/gui-preview.png");
                QTimer::singleShot(0, qApp, [ok] { qApp->exit(ok ? 0 : 1); });
            }
    }

    void start() {
        if (running) return;
        running = true;
        run_button->setEnabled(false);
        open->setEnabled(false); pattern->setEnabled(false);
        save_button->setEnabled(false);
        status->setText("Processing on GPU...");
        const QStringList source_paths = paths;
        const int source_revision = revision;
        const int width = width_box->value(), height = height_box->value(), batch = batch_box->value();
        const int method = method_box->currentIndex();
        const bool preprocess = mode_box->currentIndex() == 1;
        QMetaObject::invokeMethod(worker, [this, source_paths, source_revision, width, height, batch, method, preprocess] {
            auto completed = worker->process(source_paths, source_revision, width, height, batch, method, preprocess);
            QMetaObject::invokeMethod(this, [this, completed = std::move(completed)]() mutable {
                finish(std::move(completed));
            }, Qt::QueuedConnection);
        }, Qt::QueuedConnection);
    }
    void smokeTest(QString directory) {
        smoke = true; smoke_dir = directory; batch_box->setValue(3); mode_box->setCurrentIndex(1);
        QTimer::singleShot(0, this, &ResizeWindow::start);
    }

protected:
    void closeEvent(QCloseEvent* event) override {
        if (running) { status->setText("Wait for the current GPU operation before closing."); event->ignore(); }
        else event->accept();
    }
    void resizeEvent(QResizeEvent* event) override { QWidget::resizeEvent(event); refresh(); }

private:
    void refresh() {
        const int n = slice_box->currentIndex();
        if (n >= 0 && n < static_cast<int>(result.sources.size()))
            source_view->setPixmap(QPixmap::fromImage(result.sources[n]).scaled(source_view->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        if (n >= 0 && n < static_cast<int>(result.images.size()))
            output_view->setPixmap(QPixmap::fromImage(result.images[n]).scaled(output_view->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    QStringList paths;
    QString smoke_dir;
    int revision = 0;
    bool smoke = false;
    int smoke_run = 0;
    Result result;
    QThread worker_thread;
    BatchWorker* worker = nullptr;
    bool running = false;
    QLabel *source_view, *output_view, *status, *path_label;
    QPushButton *run_button, *save_button, *open, *pattern;
    QSpinBox *width_box, *height_box, *batch_box;
    QComboBox *method_box, *mode_box, *slice_box;
};

// Exercise real file batches, cache invalidation, resize changes and failure recovery.
int testFileBatches(const QString& directory) {
    try {
        BatchWorker worker;
        QStringList files;
        for (int n = 0; n < 3; ++n) {
            QImage image(19, 13, QImage::Format_RGB888);
            image.fill(QColor(20 + n * 70, 40 + n * 30, 200 - n * 40));
            const QString path = directory + QString("/batch-input-%1.png").arg(n);
            if (!image.save(path)) throw std::runtime_error("Cannot save test input");
            files.push_back(path);
        }
        auto verify = [](const Result& result, int width, int height, bool reused, int red = 20) {
            if (!result.error.isEmpty()) throw std::runtime_error(result.error.toStdString());
            if (result.images.size() != 3 || result.reused_input != reused)
                throw std::runtime_error("Batch count or input reuse mismatch");
            for (int n = 0; n < 3; ++n) {
                const auto& image = result.images[n];
                if (image.width() != width || image.height() != height) throw std::runtime_error("Output shape mismatch");
                const QColor expected(n == 0 ? red : 20 + n * 70, 40 + n * 30, 200 - n * 40);
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x)
                        if (image.pixelColor(x, y) != expected) throw std::runtime_error("Batch pixels mixed or stale");
            }
        };
        verify(worker.process(files, 1, 7, 5, 3, 1, false), 7, 5, false);
        verify(worker.process(files, 1, 7, 5, 3, 2, false), 7, 5, true);
        verify(worker.process(files, 1, 11, 9, 3, 1, true), 11, 9, true);
        verify(worker.process(files, 1, 19, 13, 3, 1, false), 19, 13, true);
        QImage replacement(19, 13, QImage::Format_RGB888);
        replacement.fill(QColor(33, 40, 200));
        if (!replacement.save(files[0])) throw std::runtime_error("Cannot update input");
        verify(worker.process(files, 2, 7, 5, 3, 1, false), 7, 5, false, 33);
        QImage mismatch(3, 2, QImage::Format_RGB888);
        mismatch.fill(Qt::black);
        const QString mismatch_path = directory + "/batch-mismatch.png";
        if (!mismatch.save(mismatch_path)) throw std::runtime_error("Cannot save mismatch input");
        auto invalid_files = files;
        invalid_files[1] = mismatch_path;
        if (worker.process(invalid_files, 3, 7, 5, 3, 1, false).error.isEmpty())
            throw std::runtime_error("Mismatched dimensions accepted");
        verify(worker.process(files, 4, 7, 5, 3, 1, false), 7, 5, false, 33);
        QTextStream(stdout) << "PASS: real file batch=3, distinct pixels, cached repeat, output resize, refreshed input, mismatch rejection and recovery\n";
        return 0;
    } catch (const std::exception& error) {
        QTextStream(stderr) << error.what() << '\n';
        return 1;
    }
}
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    // Offscreen Windows platforms may not enumerate system fonts automatically.
    const int font_id = QFontDatabase::addApplicationFont(qEnvironmentVariable("WINDIR") + "/Fonts/segoeui.ttf");
    const auto families = QFontDatabase::applicationFontFamilies(font_id);
    if (!families.isEmpty()) app.setFont(QFont(families.front(), 10));
    const auto args = app.arguments();
    const int test_index = args.indexOf("--batch-test");
    if (test_index >= 0 && test_index + 1 < args.size()) return testFileBatches(args[test_index + 1]);
    ResizeWindow window;
    window.show();
    const int smoke_index = args.indexOf("--smoke-test");
    if (smoke_index >= 0 && smoke_index + 1 < args.size()) window.smokeTest(args[smoke_index + 1]);
    return app.exec();
}
