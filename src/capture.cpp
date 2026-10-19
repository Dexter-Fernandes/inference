#include "capture.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <stdexcept>
#include <thread>

namespace
{
constexpr std::chrono::milliseconds kReconnectInterval{500};
constexpr std::chrono::seconds kReconnectBudget{5};
// FFmpeg's default is 30 s for both: a dead stream would stall the pipeline that long before a
// reconnect starts, and each reopen attempt could block past the whole budget.
constexpr int kStreamOpenTimeoutMs = 2000;
constexpr int kStreamReadTimeoutMs = 1000;
} // namespace

bool is_camera_source(const std::string &source)
{
    return !source.empty() && source.find_first_not_of("0123456789") == std::string::npos;
}

bool is_live_source(const std::string &source)
{
    return is_camera_source(source) || source.find("://") != std::string::npos;
}

Capture::Capture(const std::string &source)
    : source_(source), is_camera_(is_camera_source(source)), is_live_(is_live_source(source))
{
    if (!open())
        throw std::runtime_error("Failed to open source: " + source_);
}

bool Capture::open()
{
    if (is_camera_)
        return cap_.open(std::stoi(source_));
    if (is_live_)
        return cap_.open(source_, cv::CAP_FFMPEG,
                         {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, kStreamOpenTimeoutMs, cv::CAP_PROP_READ_TIMEOUT_MSEC,
                          kStreamReadTimeoutMs});
    return cap_.open(source_);
}

bool Capture::read(cv::Mat &frame)
{
    if (!cap_.read(frame) && !(is_live_ && reconnect(frame)))
        return false;
    if (is_camera_)
        cv::flip(frame, frame, 1);
    return true;
}

bool Capture::reconnect(cv::Mat &frame)
{
    cap_.release();
    // A deadline, not an attempt count: a stream's reopen can take up to its open timeout.
    const auto deadline = std::chrono::steady_clock::now() + kReconnectBudget;
    while (std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(kReconnectInterval);
        if (open() && cap_.read(frame))
        {
            reconnects_++;
            return true;
        }
        cap_.release();
    }
    return false;
}
