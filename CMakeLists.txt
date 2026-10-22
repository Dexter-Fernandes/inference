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
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavcodec libavformat libavutil libswscale)

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
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE)
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.9
    GIT_SHALLOW TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE) # the image has the X11 dev libraries only
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cli11 nvtx googletest glfw imgui)
find_package(OpenGL REQUIRED)

# ImGui ships no CMake; the window code links it, the tests do not.
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

add_library(inference_core STATIC
    src/capture.cpp src/report.cpp src/preprocess.cpp src/engine.cpp src/postprocess.cpp src/render.cpp src/filter.cu
    src/pipeline.cpp src/encoder.cpp src/det_ring.cpp src/legend.cpp src/seven_segment.cpp)
target_include_directories(inference_core PUBLIC src)
target_link_libraries(inference_core PUBLIC ${OpenCV_LIBS} nvtx3-cpp nvinfer
    CUDA::cudart CUDA::nppc CUDA::nppidei CUDA::nppial)
target_link_libraries(inference_core PRIVATE PkgConfig::LIBAV rt) # glibc 2.31 keeps shm_open in librt

add_executable(inference src/main.cpp src/gui.cpp)
target_link_libraries(inference PRIVATE inference_core CLI11::CLI11 imgui)

# Test source for live-URL input: streams a file over UDP at its own frame rate.
add_executable(udp_send tools/udp_send.cpp)
target_link_libraries(udp_send PRIVATE ${OpenCV_LIBS})

# Reads the detections `inference --publish` puts in shared memory, from another process.
add_executable(det_sub tools/det_sub.cpp)
target_link_libraries(det_sub PRIVATE inference_core CLI11::CLI11)

# unit_tests needs no GPU; gpu_tests needs the GPU plus assets/ (missing assets skip).
# tests/tsan.sh runs the queue, pool and det ring tests under ThreadSanitizer.
enable_testing()
include(GoogleTest)

add_executable(unit_tests tests/test_letterbox.cpp tests/test_postprocess.cpp tests/test_capture.cpp tests/test_report.cpp
    tests/test_bounded_queue.cpp tests/test_pool.cpp tests/test_encoder.cpp tests/test_det_ring.cpp tests/test_legend.cpp tests/test_seven_segment.cpp)
target_link_libraries(unit_tests PRIVATE inference_core GTest::gtest_main)
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)

add_executable(gpu_tests tests/test_gpu.cpp)
target_link_libraries(gpu_tests PRIVATE inference_core GTest::gtest_main)
target_compile_definitions(gpu_tests PRIVATE ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")
gtest_discover_tests(gpu_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS gpu)
