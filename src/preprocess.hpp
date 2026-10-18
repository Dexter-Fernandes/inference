#pragma once

#include "common.hpp"
#include "cuda_check.hpp"
#include "engine.hpp"

#include <opencv2/core.hpp>

// Fills lb and returns the resized (unpadded) size.
cv::Size letterbox_info(cv::Size src, int size, LetterboxInfo &lb);

void letterbox(const cv::Mat &src, cv::Mat &resized, cv::Mat &out, int size, LetterboxInfo &lb);

void preprocess(FrameData &framedata, int size);

struct NppPre
{
    NppStreamContext ctx{};
    int size = 0;
    cv::Mat resized; // host, reused across frames
    DeviceBuffer<Npp8u> d_resized, d_box;
    DeviceBuffer<Npp32f> d_f32;

    explicit NppPre(cudaStream_t stream);

    // No-op unless the input size changes, so in practice runs once.
    void ensure(int sz);
};

// Resize on the CPU, upload the small image, then pad/convert/normalise on the GPU
// straight into the engine's input buffer.
void preprocess_npp(FrameData &framedata, int size, NppPre &npp, Trt &trt);

void preprocess_cpu(FrameData &framedata, int size, Trt &trt);
