#pragma once

#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

enum class StopReason
{
    Eof,
    Frames,
    Signal,
    Key
};

struct LatencyMs
{
    double p50 = 0.0, p95 = 0.0, p99 = 0.0, max = 0.0;
};

// Nearest-rank percentiles; all zero when ms is empty.
LatencyMs summarize_latency(std::vector<double> ms);

struct EncodeStats
{
    std::string codec; // libav encoder name, e.g. "libx264"
    int frames = 0;
    int dropped = 0;   // frames skipped because every encoder slot was busy
    double ms = 0.0;   // average convert + encode + mux time per frame since the warmup frame
};

struct RunReport
{
    std::string source;
    bool npp = true;
    int frames = 0;          // all frames processed
    int measured_frames = 0; // frames after the warmup reset
    double wall_s = 0.0;     // wall time of the measured frames
    double read_ms = 0.0, pre_ms = 0.0, infer_ms = 0.0, post_ms = 0.0, render_ms = 0.0;
    int reconnects = 0;
    int dropped = 0;    // live frames skipped because the pipeline was full
    LatencyMs latency;  // capture to sink, over the measured frames
    std::optional<EncodeStats> encode;
    StopReason stopped_by = StopReason::Eof;
};

// Escapes backslash and double quote, which is all a path or URL needs.
std::string json_escape(const std::string &s);

// One line of JSON plus a newline.
void write_json(std::ostream &os, const RunReport &r);
