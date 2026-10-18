#pragma once

#include "bounded_queue.hpp"
#include "capture.hpp"
#include "common.hpp"
#include "engine.hpp"
#include "pool.hpp"

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <vector>

constexpr int kWarmupFrame = 10; // each stage resets its timer when this frame reaches it

// capture -> preprocess -> infer -> post+render -> sink, one std::jthread per stage and the
// sink on the caller's thread. Each thread owns its resources (Capture, NppPre and its stream,
// the TensorRT context); frames move between them through queues and come from a fixed pool.
class Pipeline
{
public:
    // Allocates pool_size frames, which bounds the frames in flight and so the latency. From run()
    // on, only the stage threads touch trt and cap.
    Pipeline(Trt &trt, Capture &cap, const Config &cfg, bool use_npp, std::vector<std::string> names, int pool_size = 2);

    // Call once. Calls sink for every frame in capture order. When sink returns false, capture
    // stops and frames still in flight are drained without reaching sink. Returns after every
    // stage thread has joined, rethrowing the first exception a stage or sink threw.
    void run(const std::function<bool(Frame &)> &sink);

    // Live frames skipped because every pooled frame was in use. Read after run().
    int dropped() const { return dropped_; }

private:
    using FramePtr = std::shared_ptr<Frame>;
    using Queue = BoundedQueue<FramePtr>;

    void run_capture(std::stop_token graceful, Queue &out);
    void run_preprocess(Queue &in, Queue &out);
    void run_infer(Queue &in, Queue &out);
    void run_post(Queue &in, Queue &out);
    void fail(std::exception_ptr e);

    Trt &trt_;
    Capture &cap_;
    const Config cfg_;
    const bool use_npp_;
    const std::vector<std::string> names_;
    const int pool_size_;

    Pool<Frame> pool_; // outlives run()'s threads and queues, so every handle comes back
    std::stop_source abort_;
    std::mutex error_mutex_;
    std::exception_ptr error_;
    int dropped_ = 0; // capture thread only
};
