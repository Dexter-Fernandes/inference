#include "capture.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <stdexcept>
#include <thread>

namespace
{
constexpr std::chrono::milliseconds kReconnectInterval{500};
constexpr std::chrono::seconds kReconnectBudget{5};
constexpr int kReconnectAttempts = kReconnectBudget / kReconnectInterval;
} // namespace

bool is_camera_source(const std::string &source)
{
    return !source.empty() && source.find_first_not_of("0123456789") == std::string::npos;
}

Capture::Capture(const std::string &source) : source_(source), is_camera_(is_camera_source(source))
{
    if (is_camera_)
        cap_.open(std::stoi(source_));
    else
        cap_.open(source_);
    if (!cap_.isOpened())
        throw std::runtime_error("Failed to open source: " + source_);
}

bool Capture::read(cv::Mat &frame)
{
    if (!cap_.read(frame) && !(is_camera_ && reconnect(frame)))
        return false;
    if (is_camera_)
        cv::flip(frame, frame, 1);
    return true;
}

bool Capture::reconnect(cv::Mat &frame)
{
    cap_.release();
    for (int i = 0; i < kReconnectAttempts; i++)
    {
        std::this_thread::sleep_for(kReconnectInterval);
        if (cap_.open(std::stoi(source_)) && cap_.read(frame))
        {
            reconnects_++;
            return true;
        }
        cap_.release();
    }
    return false;
}
