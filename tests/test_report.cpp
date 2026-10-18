#include "report.hpp"

#include <gtest/gtest.h>

#include <sstream>

TEST(JsonEscape, BackslashAndQuote)
{
    EXPECT_EQ(json_escape(R"(C:\vid "a".mp4)"), R"(C:\\vid \"a\".mp4)");
    EXPECT_EQ(json_escape("plain"), "plain");
}

TEST(WriteJson, ExactOutput)
{
    RunReport r;
    r.source = R"(a"b\c)";
    r.npp = false;
    r.frames = 110;
    r.measured_frames = 100;
    r.wall_s = 4.0;
    r.read_ms = 1.5;
    r.pre_ms = 2.0;
    r.infer_ms = 30.0;
    r.post_ms = 0.25;
    r.render_ms = 3.0;
    r.reconnects = 2;
    r.stopped_by = StopReason::Signal;
    std::ostringstream os;
    write_json(os, r);
    EXPECT_EQ(os.str(),
              R"({"source": "a\"b\\c", "npp": false, "frames": 110, "measured_frames": 100, )"
              R"("wall_s": 4.000, "fps": 25.000, "stage_ms": {"read": 1.500, "pre": 2.000, )"
              R"("infer": 30.000, "post": 0.250, "render": 3.000}, "reconnects": 2, "stopped_by": "signal"})"
              "\n");
}

TEST(WriteJson, NoMeasuredFramesGivesZeroFpsNotInf)
{
    RunReport r;
    r.frames = 3;
    std::ostringstream os;
    write_json(os, r);
    EXPECT_NE(os.str().find("\"fps\": 0.000"), std::string::npos);
    EXPECT_NE(os.str().find("\"stopped_by\": \"eof\""), std::string::npos);
}
