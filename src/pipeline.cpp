#include "pipeline.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include "render.hpp"

#include <nvtx3/nvtx3.hpp>

#include <chrono>
#include <thread>
#include <utility>

namespace
{
using Clock = std::chrono::steady_clock;

std::vector<Frame> make_frames(const Trt &trt, int n)
{
    std::vector<Frame> frames;
    for (int i = 0; i < n; ++i)
        frames.push_back(make_frame(trt));
    return frames;
}

// Runs work under an NVTX range and tm, restarting tm at the warmup frame, and returns tm's average.
template <class Work>
double timed(cv::TickMeter &tm, const Frame &f, const char *range, Work &&work)
{
    if (f.id == kWarmupFrame)
        tm.reset();
    nvtx3::scoped_range r{range};
    tm.start();
    work();
    tm.stop();
    return tm.getAvgTimeMilli();
}
} // namespace

Pipeline::Pipeline(Trt &trt, Capture &cap, const Config &cfg, bool use_npp, std::vector<std::string> names)
    : trt_(trt), cap_(cap), cfg_(cfg), use_npp_(use_npp), names_(std::move(names)), pool_(make_frames(trt, kPoolSize))
{
}

void Pipeline::run(const std::function<bool(Frame &)> &sink)
{
    const std::stop_token abort = abort_.get_token();
    // Each queue can hold every pooled frame, so pushes never wait; the pool is the only brake.
    Queue captured(kPoolSize), preprocessed(kPoolSize), inferred(kPoolSize), done(kPoolSize);

    // Records an exception and aborts the run, so no thread is left waiting on a dead one.
    auto guarded = [this](auto body) {
        return [this, body](std::stop_token st) {
            try
            {
                body(st);
            }
            catch (...)
            {
                fail(std::current_exception());
            }
        };
    };

    std::vector<std::jthread> threads; // after the queues, so it joins before they are destroyed
    try
    {
        threads.emplace_back(guarded([&](std::stop_token graceful) { run_capture(graceful, captured); }));
        threads.emplace_back(guarded([&](std::stop_token) { run_preprocess(captured, preprocessed); }));
        threads.emplace_back(guarded([&](std::stop_token) { run_infer(preprocessed, inferred); }));
        threads.emplace_back(guarded([&](std::stop_token) { run_post(inferred, done); }));

        bool wanted = true;
        while (auto f = done.pop(abort))
            if (wanted && !sink(**f))
            {
                wanted = false;
                threads.front().request_stop(); // capture closes its queue; the rest drain behind it
            }
    }
    catch (...)
    {
        fail(std::current_exception());
    }
    threads.clear();

    if (error_)
        std::rethrow_exception(error_);
}

void Pipeline::fail(std::exception_ptr e)
{
    {
        std::lock_guard lk(error_mutex_);
        if (!error_)
            error_ = std::move(e);
    }
    abort_.request_stop();
}

void Pipeline::run_capture(std::stop_token graceful, Queue &out)
{
    const std::stop_token abort = abort_.get_token();
    cv::TickMeter tm;
    cv::Mat scratch; // a live frame with no pooled frame free is read here and dropped
    for (int id = 1; !graceful.stop_requested();)
    {
        // A file waits for a free frame. A camera keeps reading so the next frame is current.
        FramePtr f = cap_.is_camera() ? pool_.try_acquire() : pool_.acquire(abort);
        if (abort.stop_requested())
            return;
        if (!f)
        {
            if (!cap_.read(scratch))
                break;
            ++dropped_;
            continue;
        }
        f->id = id;
        bool ok = false;
        f->stage_ms.read = timed(tm, *f, "read", [&] { ok = cap_.read(f->image); });
        if (!ok)
            break;
        f->captured = Clock::now();
        if (!out.push(std::move(f), abort))
            return;
        ++id;
    }
    out.close();
}

void Pipeline::run_preprocess(Queue &in, Queue &out)
{
    const std::stop_token abort = abort_.get_token();
    CudaStream stream; // not the TensorRT stream, so this overlaps inference of the previous frame
    NppPre npp(stream.get());
    cv::TickMeter tm;
    while (auto f = in.pop(abort))
    {
        Frame &fr = **f;
        fr.stage_ms.pre = timed(tm, fr, "pre", [&] {
            if (use_npp_)
                preprocess_npp(fr, cfg_.input_size, npp);
            else
                preprocess_cpu(fr, cfg_.input_size, stream.get());
        });
        if (!out.push(std::move(*f), abort))
            return;
    }
    out.close();
}

void Pipeline::run_infer(Queue &in, Queue &out)
{
    const std::stop_token abort = abort_.get_token();
    cv::TickMeter tm;
    while (auto f = in.pop(abort))
    {
        Frame &fr = **f;
        fr.stage_ms.infer = timed(tm, fr, "infer", [&] { infer(trt_, fr, cfg_); });
        if (!out.push(std::move(*f), abort))
            return;
    }
    out.close();
}

void Pipeline::run_post(Queue &in, Queue &out)
{
    const std::stop_token abort = abort_.get_token();
    cv::TickMeter post_tm, render_tm;
    Clock::time_point last{};
    double fps_ema = 0.0;
    while (auto f = in.pop(abort))
    {
        Frame &fr = **f;
        fr.stage_ms.post = timed(post_tm, fr, "post", [&] { postprocess_stage(fr, cfg_); });

        const Clock::time_point now = Clock::now();
        if (last != Clock::time_point{})
        {
            const double fps = 1.0 / std::chrono::duration<double>(now - last).count();
            fps_ema = fps_ema == 0.0 ? fps : 0.9 * fps_ema + 0.1 * fps;
        }
        last = now;

        fr.stage_ms.render = timed(render_tm, fr, "render", [&] { render(fr, names_, fps_ema); });
        if (!out.push(std::move(*f), abort))
            return;
    }
    out.close();
}
