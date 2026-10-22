#pragma once

#include "common.hpp"
#include "legend.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <vector>

// BGR, chosen to stay apart on a video; the sidebar legend and the boxes share it.
const std::array<cv::Scalar, kPaletteSize> &palette();

// Boxes only. A class for which color_of returns nullopt is skipped.
void draw_boxes(cv::Mat &frame, const std::vector<Detection> &dets,
                const std::function<std::optional<cv::Scalar>(int class_id)> &color_of);

// Headless path: boxes with "name conf" labels, plus an FPS/Det HUD, burned into frame.image.
void render(Frame &frame, const std::vector<std::string> &names);
