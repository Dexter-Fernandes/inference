cmake_minimum_required(VERSION 3.16)
project(inference LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimised build with debug symbols
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.9")
set(CUDAToolkit_ROOT "/usr/local/cuda-12.9")
find_package(CUDAToolkit REQUIRED)
list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local/opencv-cuda")
cmake_policy(SET CMP0146 OLD)
find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.7.2)
FetchContent_MakeAvailable(cli11)

add_executable(inference main.cpp)
target_link_libraries(inference PRIVATE ${OpenCV_LIBS} CLI11::CLI11
    CUDA::cudart CUDA::nppc CUDA::nppig CUDA::nppidei CUDA::nppial)
