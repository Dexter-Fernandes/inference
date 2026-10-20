#pragma once

#include "capture.hpp"
#include "report.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

// Writes frames to a file or a stream with a CPU encoder on its own thread, so the pipeline never
// waits for it.
class Encoder
{
public:
    // The libav encoder names accepted as `codec`.
    static const std::vector<std::string> &codec_names();

    // Opens the codec and the output and writes the container header, so a bad target, codec or
    // container/codec pair throws here, before any frame. target is a file (container from its
    // extension) or a udp:// URL (always MPEG-TS). The output is fmt.size rounded down to even and
    // fmt.fps; a frame of another size is scaled to it. slots bounds the frames waiting or being
    // encoded. Throws std::invalid_argument for an unknown codec or an empty size, else
    // std::runtime_error naming the libav error.
    Encoder(const std::string &target, const std::string &codec, const SourceFormat &fmt, int slots = 2);

    // Drains at most `slots` frames and writes the trailer best-effort, so a partial file still
    // plays. Never throws.
    ~Encoder();

    Encoder(const Encoder &) = delete;
    Encoder &operator=(const Encoder &) = delete;

    // Call from one thread. Deep-copies bgr (CV_8UC3), so the caller may overwrite it as soon as
    // this returns. Never blocks: if every slot is busy the frame is dropped and counted. Rethrows the
    // encoder thread's error. id and captured are Frame::id and Frame::captured.
    void submit(const cv::Mat &bgr, int id, std::chrono::steady_clock::time_point captured);

    // Drains the queue, flushes the codec, writes the trailer and joins, then returns the run's
    // numbers or rethrows the encoder thread's error. Idempotent. No submit after it.
    [[nodiscard]] EncodeStats finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
