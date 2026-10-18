cmake_minimum_required(VERSION 3.16)
project(inference LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 50) # GeForce 940MX

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17) # nvcc 12.1 + GCC 9 can't do C++20

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

add_executable(inference main.cpp filter.cu)
target_link_libraries(inference PRIVATE ${OpenCV_LIBS} CLI11::CLI11 nvtx3-cpp nvinfer
    CUDA::cudart CUDA::nppc CUDA::nppidei CUDA::nppial)
