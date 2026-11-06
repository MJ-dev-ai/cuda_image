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

set_target_properties(cuimg PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(MSVC)
    target_compile_options(cuimg PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

# Use only the project-local Qt SDK; runtime files stay in externals.
set(QT_SDK "${CMAKE_CURRENT_SOURCE_DIR}/externals/Qt/6.8.3/msvc2022_64" CACHE PATH "Local Qt SDK")
find_package(Qt6 REQUIRED COMPONENTS Widgets PATHS "${QT_SDK}/lib/cmake/Qt6" NO_DEFAULT_PATH)
get_filename_component(PROJECT_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
string(REGEX REPLACE "^[0-9][0-9]_" "" APP_NAME "${PROJECT_FOLDER}")
add_executable(resize_gui src/resize_test_gui.cpp)
target_link_libraries(resize_gui PRIVATE cx::cuimg Qt6::Widgets)
set_target_properties(resize_gui PROPERTIES OUTPUT_NAME "${APP_NAME}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(CONFIG DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(resize_gui PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${CONFIG} "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()

enable_testing()
add_executable(resize_test tests/resize_test.cpp)
target_link_libraries(resize_test PRIVATE cx::cuimg)
set_target_properties(resize_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_test(NAME resize_pixels COMMAND resize_test)
