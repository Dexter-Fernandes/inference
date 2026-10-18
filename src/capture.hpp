#pragma once

#include "common.hpp"

#include <opencv2/videoio.hpp>

#include <string>
#include <utility>

// Camera if source is all digits, otherwise a file or URL. The capture may be unopened.
std::pair<cv::VideoCapture, bool> open_source(const std::string &source);

bool read_frame(cv::VideoCapture &cap, bool flip, FrameData &framedata);
