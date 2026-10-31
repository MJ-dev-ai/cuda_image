cmake_minimum_required(VERSION 3.24)

project(cuimg VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(cuimg STATIC
    src/cuimg/cuda_tensor_buffer.cpp
    src/cuimg/image_buffer.cpp
    src/cuimg/image_io.cpp
    src/cuimg/interpolation.cu
    src/cuimg/preprocess.cu
    src/cuimg/stb.cpp
)
add_library(cx::cuimg ALIAS cuimg)

target_compile_features(cuimg PUBLIC cxx_std_17)
set_target_properties(cuimg PROPERTIES
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    # Allow kernels in other translation units to call interpolation device code.
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(cuimg
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_include_directories(cuimg SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/externals"
)

# Public headers expose CUDA types, so consumers need CUDA include paths too.
target_link_libraries(cuimg PUBLIC CUDA::cudart)
