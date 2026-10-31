#include "cuimg/preprocess.hpp"
#include <stdexcept>

namespace cx {

std::vector<LetterboxTransform> preprocessYolo(const ImageBuffer&, const TensorView&, cudaStream_t) {
    // TODO: Validate input format, tensor capacity and dimensions, then launch kernel.
    throw std::logic_error("cx::preprocessYolo is not implemented");
}


__device__ bool nhwc2nchw(
    const unsigned char* src, unsigned char* dst, 
    int batch, int height, int width, int channels) {
        
}
} // namespace cx
