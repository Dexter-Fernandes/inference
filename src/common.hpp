#pragma once

#include "cuda_check.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <chrono>
#include <set>
#include <vector>

struct Config
{
    int input_size = 640;
    float conf_thr = 0.25f;
    float nms_thr = 0.5f;
    std::set<int> keep_classes; // empty = keep all
};

struct Detection
{
    int class_id;
    float conf;
    cv::Rect box;
};

struct LetterboxInfo
{
    float scale;
    int pad_x, pad_y;
};

// Average ms per stage since the warmup frame.
struct StageMs
{
    double read = 0, pre = 0, infer = 0, post = 0, render = 0;
};

// One video frame on its way through the stages. d_in and out are allocated once (make_frame)
// and reused, so a frame owns its engine input and nothing is shared between frames in flight.
struct Frame
{
    int id = 0; // 1-based: capture order until infer renumbers it to the order frames reach the sink
    std::chrono::steady_clock::time_point captured;
    cv::Mat image; // original, drawn on at the end
    LetterboxInfo lb{};
    cv::Mat blob;             // NCHW input to the net, CPU preprocess only
    DeviceBuffer<float> d_in; // engine input
    cv::Mat out;              // net output rows that passed the GPU objectness filter
    int num_kept = 0;         // valid rows at the start of out
    std::vector<Detection> dets;
    double fps = 0; // smoothed throughput when the post thread handled this frame
    StageMs stage_ms; // each stage writes its own running average as the frame passes
};
