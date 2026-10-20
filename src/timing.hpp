#pragma once

#include <opencv2/core/utility.hpp>

#include <nvtx3/nvtx3.hpp>

constexpr int kWarmupFrame = 10; // each stage resets its timer when this frame reaches it

// Runs work under an NVTX range and tm, restarting tm at the warmup frame, and returns tm's average.
template <class Work>
double timed(cv::TickMeter &tm, int frame_id, const char *range, Work &&work)
{
    if (frame_id == kWarmupFrame)
        tm.reset();
    nvtx3::scoped_range r{range};
    tm.start();
    work();
    tm.stop();
    return tm.getAvgTimeMilli();
}
