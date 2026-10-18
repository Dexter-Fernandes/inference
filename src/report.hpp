#pragma once

#include <iosfwd>
#include <string>

enum class StopReason
{
    Eof,
    Frames,
    Signal,
    Key
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
    StopReason stopped_by = StopReason::Eof;
};

// Escapes backslash and double quote, which is all a path or URL needs.
std::string json_escape(const std::string &s);

// One line of JSON plus a newline.
void write_json(std::ostream &os, const RunReport &r);
