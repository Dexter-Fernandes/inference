cmake_minimum_required(VERSION 3.16)
project(inference LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 50) # GeForce 940MX

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17) # CMake 3.24 has no CUDA20 dialect for nvcc

# Optimised build with debug symbols
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Built inside the TensorRT container (see Dockerfile / compose.yaml)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.7.2)
# CUDA 12.1 ships only the NVTX C headers, not nvtx3.hpp
FetchContent_Declare(nvtx
    GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR c)
FetchContent_MakeAvailable(cli11 nvtx)

add_library(inference_core STATIC
    src/capture.cpp src/preprocess.cpp src/engine.cpp src/postprocess.cpp src/render.cpp src/filter.cu)
target_include_directories(inference_core PUBLIC src)
target_link_libraries(inference_core PUBLIC ${OpenCV_LIBS} nvtx3-cpp nvinfer
    CUDA::cudart CUDA::nppc CUDA::nppidei CUDA::nppial)

add_executable(inference src/main.cpp)
target_link_libraries(inference PRIVATE inference_core CLI11::CLI11)
