#pragma once

#include "common.hpp"

#include <opencv2/core.hpp>

#include <vector>

std::vector<Detection> postprocess(const cv::Mat &out, int num_boxes, const LetterboxInfo &lb, const cv::Size &orig, const Config &cfg);

void postprocess_stage(Frame &frame, const Config &config);
