#include "capture.hpp"
#include "common.hpp"
#include "det_ring.hpp"
#include "encoder.hpp"
#include "engine.hpp"
#include "gui.hpp"
#include "pipeline.hpp"
#include "report.hpp"
#include "stop_signal.hpp"

#include <CLI/CLI.hpp>

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{
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

    std::optional<int> pool_arg;
    app.add_option("--pool", pool_arg,
                   "Frames in flight; each one adds about one inference time of latency (default: 3 for a live source, 2 for a file)")
        ->check(CLI::Range(2, 16));

    bool headless = false;
    app.add_flag("--headless", headless, "No window");

    std::string json_path;
    app.add_option("--json", json_path, "Write a JSON run report to PATH at the end ('-' for stdout)");

    std::string encode_target;
    auto *encode_opt = app.add_option("--encode", encode_target,
                                      "Also write the rendered video to FILE (container from the extension) or udp://host:port (MPEG-TS)");
    std::string codec = "libx264";
    app.add_option("--codec", codec, "Encoder for --encode")
        ->check(CLI::IsMember(Encoder::codec_names()))
        ->needs(encode_opt)
        ->capture_default_str();

    std::string publish_name = "/inference_dets";
    auto *publish_opt = app.add_option("--publish", publish_name,
                                       "Also publish each frame's detections to shared memory NAME for tools/det_sub")
                            ->expected(0, 1)
                            ->capture_default_str();

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

    std::optional<Encoder> encoder;
    if (!encode_target.empty())
    {
        // The encoder truncates its target on open, and capture already holds the source.
        std::error_code ec;
        if (!cap.is_live() && std::filesystem::equivalent(source, encode_target, ec))
            throw std::runtime_error("--encode would overwrite the --source file: " + source);
        encoder.emplace(encode_target, codec, cap.format());
    }

    std::vector<std::string> names; // empty is fine: the labels and the legend fall back to class ids
    {
        std::ifstream f("assets/coco.names");
        if (!f)
            std::cerr << "Warning: assets/coco.names not found, showing class ids\n";
        for (std::string line; std::getline(f, line);)
            names.push_back(line);
    }

    // A live source needs the third frame to hold a fresh one beside the frames in infer and waiting; a file
    // blocks, so a third frame would only add latency.
    const int pool_size = pool_arg.value_or(cap.is_live() ? 3 : 2);
    std::optional<DetWriter> publisher;
    if (publish_opt->count() > 0)
        publisher.emplace(publish_name);
    // The window draws its own boxes in the sink, so the post thread annotates only for a headless run.
    std::optional<Gui> gui;
    if (!headless)
        gui.emplace(names);
    Pipeline pipeline(trt, cap, config, use_npp, std::move(names), pool_size, publisher ? &*publisher : nullptr, headless);
    install_signal_handlers();

    using Clock = std::chrono::steady_clock;
    int frames = 0;
    StageMs stage_ms;
    std::vector<double> latency_ms;
    Clock::time_point measure_start, measure_end;
    StopReason stopped_by = StopReason::Eof;
    pipeline.run([&](Frame &f) {
        const bool open = !gui || gui->show(f);

        const Clock::time_point now = Clock::now();
        frames = f.id;
        stage_ms = f.stage_ms;
        if (f.id == kWarmupFrame - 1) // throughput counts the intervals that end at frames 10..N
            measure_start = now;
        if (f.id >= kWarmupFrame)
        {
            measure_end = now;
            if (json_out) // 8 bytes a frame, only for the report
                latency_ms.push_back(std::chrono::duration<double, std::milli>(now - f.captured).count());
        }

        // After the latency stamp, so the copy is not counted; before the stop checks, so the frame
        // that ends a run is encoded too.
        if (encoder)
            encoder->submit(f.image, f.id, f.captured);

        if (f.id % 100 == 0)
        {
            std::cerr << stage_line(f.id, stage_ms) << "\n";
            if (gui)
                gui->set_stage_rows(stage_rows(f.id, stage_ms));
        }

        if (!open)
            stopped_by = StopReason::Key;
        else if (g_stop.load(std::memory_order_relaxed))
            stopped_by = StopReason::Signal;
        else if (f.id == max_frames)
            stopped_by = StopReason::Frames;
        else
            return true;
        return false;
    });

    std::optional<EncodeStats> encode_stats;
    if (encoder)
        encode_stats = encoder->finish();

    if (json_out)
    {
        RunReport report;
        report.source = source;
        report.npp = use_npp;
        report.frames = frames;
        report.reconnects = cap.reconnects();
        report.dropped = pipeline.dropped();
        report.stopped_by = stopped_by;
        if (frames >= kWarmupFrame)
        {
            report.measured_frames = frames - kWarmupFrame + 1;
            report.wall_s = std::chrono::duration<double>(measure_end - measure_start).count();
        }
        report.read_ms = stage_ms.read;
        report.pre_ms = stage_ms.pre;
        report.infer_ms = stage_ms.infer;
        report.post_ms = stage_ms.post;
        report.render_ms = stage_ms.render;
        report.latency = summarize_latency(std::move(latency_ms));
        report.encode = std::move(encode_stats);
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
