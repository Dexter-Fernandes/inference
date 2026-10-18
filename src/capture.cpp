#include "capture.hpp"

#include <opencv2/core.hpp>

std::pair<cv::VideoCapture, bool> open_source(const std::string &source)
{
    cv::VideoCapture cap;
    // is_camera if source only contains numbers
    const bool is_camera = !source.empty() && source.find_first_not_of("0123456789") == std::string::npos;
    if (is_camera)
        cap.open(std::stoi(source));
    else
        cap.open(source);
    return {std::move(cap), is_camera};
}

bool read_frame(cv::VideoCapture &cap, bool flip, FrameData &framedata)
{
    if (!cap.read(framedata.frame))
        return false;
    if (flip)
        cv::flip(framedata.frame, framedata.frame, 1);
    return true;
}
