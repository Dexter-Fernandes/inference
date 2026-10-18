#include "capture.hpp"
#include "common.hpp"
#include "engine.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include "render.hpp"
#include "report.hpp"

#include <nvtx3/nvtx3.hpp>
#include <opencv2/highgui.hpp>
#include <CLI/CLI.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
std::atomic<bool> g_stop{false};
static_assert(std::atomic<bool>::is_always_lock_free);

void on_signal(int) { g_stop.store(true, std::memory_order_relaxed); }

void install_signal_handlers()
{
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    for (const int sig : {SIGINT, SIGTERM})
        sigaction(sig, &sa, nullptr);
}

constexpr int kWarmupFrame = 10; // timers reset here

int run(int argc, char **argv)
{
    CLI::App app("Runs an object detection on a video or webcam");
    argv = app.ensure_utf8(argv);

    Config config;

    std::string model_path = "assets/yolov5s.engine";
    app.add_option("-m,--model_path", model_path, "Path to the TensorRT engine")
        ->check(CLI::ExistingFile)
        ->capture_default_str();

    std::string source = "0";
    app.add_option("-s,--source", source, "Camera index, video file or URL")->capture_default_str();

    bool use_npp = true;
    app.add_flag("--npp,!--no-npp", use_npp, "Preprocess on GPU with NPP");

    app.add_option("--conf", config.conf_thr, "Confidence threshold")
        ->check(CLI::Range(0.0, 1.0))
        ->capture_default_str();
    app.add_option("--nms", config.nms_thr, "NMS IoU threshold")
        ->check(CLI::Range(0.0, 1.0))
        ->capture_default_str();

    int max_frames = std::numeric_limits<int>::max();
    app.add_option("--frames", max_frames, "Stop after N frames (default: unlimited)")
        ->check(CLI::PositiveNumber);

    bool headless = false;
    app.add_flag("--headless", headless, "No window; nothing touches highgui");

    std::string json_path;
    app.add_option("--json", json_path, "Write a JSON run report to PATH at the end ('-' for stdout)");

    CLI11_PARSE(app, argc, argv);

    std::ofstream json_file;
    std::ostream *json_out = nullptr;
    if (json_path == "-")
        json_out = &std::cout;
    else if (!json_path.empty())
    {
        json_file.open(json_path);
        if (!json_file)
            throw std::runtime_error("Cannot write --json file: " + json_path);
        json_out = &json_file;
    }

    Trt trt(model_path);
    config.input_size = trt.in_dims.d[2]; // NCHW, square

    Capture cap(source);

    Frame fd = make_frame(trt);
    NppPre npp(trt.stream.get());

    std::vector<std::string> names; // empty is fine: draw() falls back to class ids
    {
        std::ifstream f("assets/coco.names");
        if (!f)
            std::cerr << "Warning: assets/coco.names not found, showing class ids\n";
        for (std::string line; std::getline(f, line);)
            names.push_back(line);
    }

    install_signal_handlers();

    using Clock = std::chrono::steady_clock;
    cv::TickMeter frame_tm;
    StageTimers t{};
    double fps_ema = 0.0;
    int frames = 0;
    Clock::time_point measure_start, measure_end;
    StopReason stopped_by = StopReason::Eof;
    for (int n = 1;; n++)
    {
        if (g_stop.load(std::memory_order_relaxed))
        {
            stopped_by = StopReason::Signal;
            break;
        }

        if (n == kWarmupFrame)
        {
            for (auto *tm : {&t.read, &t.pre, &t.infer, &t.post, &t.render})
                tm->reset();
            measure_start = Clock::now();
        }

        bool ret;
        {
            nvtx3::scoped_range r{"read"};
            t.read.start();
            ret = cap.read(fd.image);
            t.read.stop();
        }
        if (!ret)
            break;

        frame_tm.reset();
        frame_tm.start();

        {
            nvtx3::scoped_range r{"pre"};
            t.pre.start();
            if (use_npp)
                preprocess_npp(fd, config.input_size, npp);
            else
                preprocess_cpu(fd, config.input_size, trt.stream.get());
            t.pre.stop();
        }

        {
            nvtx3::scoped_range r{"infer"};
            t.infer.start();
            infer(trt, fd, config);
            t.infer.stop();
        }

        {
            nvtx3::scoped_range r{"post"};
            t.post.start();
            postprocess_stage(fd, config);
            t.post.stop();
        }

        frame_tm.stop();
        const double fps = 1.0 / frame_tm.getTimeSec();
        if (fps_ema == 0.0)
            fps_ema = fps;
        else
            fps_ema = 0.9 * fps_ema + 0.1 * fps;

        {
            nvtx3::scoped_range r{"render"};
            t.render.start();
            render(fd, names, fps_ema);
            if (!headless)
                cv::imshow("Playback", fd.image);
            t.render.stop();
        }

        if (n % 100 == 0)
        {
            std::cerr << cv::format("frame %d: read %.1f | pre %.1f | infer %.1f | post %.1f | render %.1f ms\n",
                                    n, t.read.getAvgTimeMilli(), t.pre.getAvgTimeMilli(), t.infer.getAvgTimeMilli(), t.post.getAvgTimeMilli(), t.render.getAvgTimeMilli());
        }

        const int key = headless ? 0 : cv::waitKey(1) & 0xFF;
        frames = n;
        measure_end = Clock::now();
        if (key == 'q' || key == 27)
        {
            stopped_by = StopReason::Key;
            break;
        }
        if (n == max_frames)
        {
            stopped_by = StopReason::Frames;
            break;
        }
    }

    if (json_out)
    {
        RunReport report;
        report.source = source;
        report.npp = use_npp;
        report.frames = frames;
        report.reconnects = cap.reconnects();
        report.stopped_by = stopped_by;
        if (frames >= kWarmupFrame)
        {
            report.measured_frames = frames - kWarmupFrame + 1;
            report.wall_s = std::chrono::duration<double>(measure_end - measure_start).count();
        }
        report.read_ms = t.read.getAvgTimeMilli();
        report.pre_ms = t.pre.getAvgTimeMilli();
        report.infer_ms = t.infer.getAvgTimeMilli();
        report.post_ms = t.post.getAvgTimeMilli();
        report.render_ms = t.render.getAvgTimeMilli();
        write_json(*json_out, report);
    }
    return 0;
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
