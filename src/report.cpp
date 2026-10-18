#include "report.hpp"

#include <iomanip>
#include <ostream>
#include <sstream>

namespace
{
const char *to_string(StopReason r)
{
    switch (r)
    {
    case StopReason::Eof:
        return "eof";
    case StopReason::Frames:
        return "frames";
    case StopReason::Signal:
        return "signal";
    case StopReason::Key:
        return "key";
    }
    return "eof";
}
} // namespace

std::string json_escape(const std::string &s)
{
    std::string out;
    for (const char c : s)
    {
        if (c == '\\' || c == '"')
            out += '\\';
        out += c;
    }
    return out;
}

void write_json(std::ostream &os, const RunReport &r)
{
    const double fps = r.wall_s > 0.0 ? r.measured_frames / r.wall_s : 0.0;
    std::ostringstream s;
    s << std::fixed << std::setprecision(3);
    s << "{\"source\": \"" << json_escape(r.source) << "\", \"npp\": " << (r.npp ? "true" : "false")
      << ", \"frames\": " << r.frames << ", \"measured_frames\": " << r.measured_frames
      << ", \"wall_s\": " << r.wall_s << ", \"fps\": " << fps
      << ", \"stage_ms\": {\"read\": " << r.read_ms << ", \"pre\": " << r.pre_ms
      << ", \"infer\": " << r.infer_ms << ", \"post\": " << r.post_ms << ", \"render\": " << r.render_ms
      << "}, \"reconnects\": " << r.reconnects << ", \"stopped_by\": \"" << to_string(r.stopped_by) << "\"}\n";
    os << s.str() << std::flush;
}
