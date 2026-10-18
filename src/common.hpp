#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

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

struct FrameData
{
    cv::Mat frame; // original, drawn on at the end
    cv::Mat blob;  // NCHW input to the net
    LetterboxInfo lb;
    cv::Mat out;      // net output rows that passed the GPU objectness filter
    int num_kept = 0; // valid rows at the start of out
    std::vector<Detection> dets;
};

struct StageTimers
{
    cv::TickMeter read, pre, infer, post, render;
};
