// Reads the detections `inference --publish` writes to shared memory and reports how long each
// frame took from publish to read. Start it after the producer; a producer restart needs a restart.
// det_sub [--name NAME] [--poll-us N] [--frames N] [--quiet]
#include "det_ring.hpp"
#include "report.hpp"
#include "stop_signal.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <limits>
#include <thread>
#include <vector>

namespace
{
int run(int argc, char **argv)
{
    CLI::App app("Prints the detections another process publishes to shared memory");

    std::string name = "/inference_dets";
    app.add_option("--name", name, "Shared memory name given to inference --publish")->capture_default_str();

    int poll_us = 100;
    app.add_option("--poll-us", poll_us, "Sleep between checks for a new frame; 0 spins")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();

    int max_frames = std::numeric_limits<int>::max();
    app.add_option("--frames", max_frames, "Stop after N frames (default: until SIGINT)")->check(CLI::PositiveNumber);

    bool quiet = false;
    app.add_flag("--quiet", quiet, "Print only the summary");

    CLI11_PARSE(app, argc, argv);

    DetReader reader(name);

    install_signal_handlers();

    using Clock = std::chrono::steady_clock;
    uint64_t next = reader.published(); // only frames published from now on
    uint64_t skipped = 0;
    int frames = 0;
    Clock::time_point last_frame = Clock::now();
    bool warned = false;
    std::vector<double> latency_us;
    DetFrame f;
    while (frames < max_frames && !g_stop.load(std::memory_order_relaxed))
    {
        switch (reader.read(next, f))
        {
        case ReadResult::Ok:
        {
            const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
            latency_us.push_back((now - f.published_ns) / 1e3);
            ++frames;
            ++next;
            last_frame = Clock::now();
            warned = false;
            if (!quiet)
            {
                std::printf("frame %llu: %u dets", static_cast<unsigned long long>(f.frame_id), f.count);
                for (uint32_t i = 0; i < f.count; ++i)
                    std::printf(" [%d %.2f %d,%d %dx%d]", f.dets[i].class_id, f.dets[i].conf, f.dets[i].x, f.dets[i].y,
                                f.dets[i].w, f.dets[i].h);
                std::printf("\n");
            }
            break;
        }
        case ReadResult::Pending:
            // A producer killed or restarted since we attached leaves us on a ring nobody writes to.
            if (!warned && Clock::now() - last_frame > std::chrono::seconds(3))
            {
                std::fprintf(stderr, "no new frame for 3 s; if the producer restarted, restart det_sub\n");
                warned = true;
            }
            if (poll_us > 0)
                std::this_thread::sleep_for(std::chrono::microseconds(poll_us));
            break;
        case ReadResult::Lapped:
        {
            // The newest complete publication is published() - 1; published() itself is still being made.
            const uint64_t head = std::max(next + 1, reader.published() - 1); // a lap means at least kRingSlots published
            skipped += head - next; // includes frames still readable: we jump to the newest on purpose
            next = head;
            break;
        }
        }
    }

    // Only ranks the values, so microseconds go in as well as milliseconds.
    const LatencyMs l = summarize_latency(std::move(latency_us));
    std::printf("%d frames, %llu skipped (lapped), publish-to-read us p50 %.1f p95 %.1f p99 %.1f max %.1f (poll %d us)\n", frames,
                static_cast<unsigned long long>(skipped), l.p50, l.p95, l.p99, l.max, poll_us);
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
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
