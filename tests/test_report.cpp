#include "report.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <vector>

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
    r.dropped = 7;
    r.latency = {40.0, 45.5, 50.25, 61.0};
    r.stopped_by = StopReason::Signal;
    std::ostringstream os;
    write_json(os, r);
    EXPECT_EQ(os.str(),
              R"({"source": "a\"b\\c", "npp": false, "frames": 110, "measured_frames": 100, )"
              R"("wall_s": 4.000, "fps": 25.000, "stage_ms": {"read": 1.500, "pre": 2.000, )"
              R"("infer": 30.000, "post": 0.250, "render": 3.000}, "reconnects": 2, "dropped": 7, )"
              R"("latency_ms": {"p50": 40.000, "p95": 45.500, "p99": 50.250, "max": 61.000}, "stopped_by": "signal"})"
              "\n");
}

TEST(WriteJson, EncodeBlockSitsBeforeStoppedBy)
{
    RunReport r;
    r.encode = EncodeStats{.codec = "libx264", .frames = 98, .dropped = 2, .ms = 12.5};
    std::ostringstream os;
    write_json(os, r);
    EXPECT_NE(os.str().find(R"("max": 0.000}, "encode": {"codec": "libx264", "frames": 98, "dropped": 2, "ms": 12.500}, )"
                            R"("stopped_by": "eof"})"
                            "\n"),
              std::string::npos)
        << os.str();
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

TEST(SummarizeLatency, NearestRankPercentiles)
{
    // 1..100 ms in scrambled order (37 is coprime to 100): the p-th percentile by nearest rank is p ms.
    std::vector<double> ms;
    for (int i = 0; i < 100; ++i)
        ms.push_back((i * 37) % 100 + 1);
    const LatencyMs l = summarize_latency(ms);
    EXPECT_EQ(l.p50, 50.0);
    EXPECT_EQ(l.p95, 95.0);
    EXPECT_EQ(l.p99, 99.0);
    EXPECT_EQ(l.max, 100.0);
}

TEST(SummarizeLatency, SmallAndEmptySamples)
{
    const LatencyMs one = summarize_latency({7.0});
    EXPECT_EQ(one.p50, 7.0);
    EXPECT_EQ(one.max, 7.0);

    const LatencyMs three = summarize_latency({30.0, 10.0, 20.0});
    EXPECT_EQ(three.p50, 20.0); // rank ceil(1.5) = 2
    EXPECT_EQ(three.p95, 30.0);

    const LatencyMs none = summarize_latency({});
    EXPECT_EQ(none.p50, 0.0);
    EXPECT_EQ(none.max, 0.0);
}
