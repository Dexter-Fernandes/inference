#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>

// A source made only of digits is a camera index.
bool is_camera_source(const std::string &source);

// A camera (mirrored, reopened when reads fail) or a video file or URL (ends at EOF).
class Capture
{
public:
    // Throws std::runtime_error if the source cannot be opened.
    explicit Capture(const std::string &source);

    // False at the end of a file, or when a camera did not come back in time.
    bool read(cv::Mat &frame);

    bool is_camera() const { return is_camera_; }
    int reconnects() const { return reconnects_; }

private:
    bool reconnect(cv::Mat &frame);

    std::string source_;
    bool is_camera_;
    cv::VideoCapture cap_;
    int reconnects_ = 0;
};
