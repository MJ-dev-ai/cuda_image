CMAKE_MINIMUM_REQUIRED_VERSION(2.8)

PROJECT(
    "CuImage"
    VERSION 1.0
    DESCRIPTION "Image processing with CUDA memory"
    LANGUAGES CXX
)

ADD_EXECUTABLE(${PROJECT_NAME}
    src/main.cpp
    src/settings.cpp
    src/core/device_buffer.cpp
    src/core/host_buffer.cpp
    src/core/image_buffer.cpp
    src/cx/interpolation.cu
    src/cx/resize.cu
)