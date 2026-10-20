#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>

// A source made only of digits is a camera index.
bool is_camera_source(const std::string &source);

// A camera or a stream URL ("scheme://..."): frames arrive whether or not we are ready for them.
bool is_live_source(const std::string &source);

// What the source reported when it opened; a reconnect does not update it.
struct SourceFormat
{
    cv::Size size; // 0x0 if the backend did not say
    double fps;    // the reported rate, or 30 when it is missing or absurd
    bool live;
};

// A live source (reopened when reads fail; a camera is also mirrored) or a video file (ends at EOF).
class Capture
{
public:
    // Throws std::runtime_error if the source cannot be opened.
    explicit Capture(const std::string &source);

    // False at the end of a file, or when a live source did not come back in time.
    bool read(cv::Mat &frame);

    bool is_live() const { return is_live_; }
    int reconnects() const { return reconnects_; }

    // Set once in the constructor, so it is safe to read once run() has handed the capture to its thread.
    const SourceFormat &format() const { return format_; }

private:
    bool open();
    bool reconnect(cv::Mat &frame);

    std::string source_;
    bool is_camera_;
    bool is_live_;
    SourceFormat format_{};
    cv::VideoCapture cap_;
    int reconnects_ = 0;
};
