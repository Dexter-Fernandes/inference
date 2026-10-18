#pragma once

#include "common.hpp"
#include "cuda_check.hpp"

#include <opencv2/core.hpp>

// Fills lb and returns the resized (unpadded) size.
cv::Size letterbox_info(cv::Size src, int size, LetterboxInfo &lb);

void letterbox(const cv::Mat &src, cv::Mat &resized, cv::Mat &out, int size, LetterboxInfo &lb);

void preprocess(Frame &frame, int size);

struct NppPre
{
    NppStreamContext ctx{};
    int size = 0;
    cv::Mat resized; // host, reused across frames
    DeviceBuffer<Npp8u> d_box;
    DeviceBuffer<Npp32f> d_f32;
    cv::Size filled_for; // resized size that d_box's 114 border was laid out for

    explicit NppPre(cudaStream_t stream);

    // No-op unless the input size changes, so in practice runs once.
    void ensure(int sz);
};

// Resize on the CPU, upload the small image into the padded box, then convert/normalise
// on the GPU straight into frame.d_in. Runs on npp.ctx.hStream and returns once it is done.
void preprocess_npp(Frame &frame, int size, NppPre &npp);

// preprocess(), then uploads the blob into frame.d_in on stream and waits for it.
void preprocess_cpu(Frame &frame, int size, cudaStream_t stream);
