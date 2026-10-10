cmake_minimum_required(VERSION 3.16)
project(inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimised build with debug symbols
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.9")
list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local/opencv-cuda")
find_package(OpenCV REQUIRED)

add_executable(inference main.cpp)
# target_include_directories(inference PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference PRIVATE ${OpenCV_LIBS})
