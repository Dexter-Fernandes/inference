#pragma once

#include "common.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

void draw(cv::Mat &frame, const std::vector<Detection> &dets,
          const std::vector<std::string> &names);

void render(FrameData &framedata, const std::vector<std::string> &names, double fps);
