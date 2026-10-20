#include "encoder.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

constexpr int kNoDropSlots = 64;

class EncoderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::string name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        std::replace(name.begin(), name.end(), '/', '_'); // a parameterized name holds slashes
        dir_ = fs::temp_directory_path() / ("encoder_test_" + name);
        fs::create_directories(dir_);
    }
    void TearDown() override { fs::remove_all(dir_); }

    std::string path(const std::string &name) const { return (dir_ / name).string(); }

    fs::path dir_;
};

cv::Mat frame(cv::Size size, int i)
{
    cv::Mat m(size, CV_8UC3, cv::Scalar(40 + i * 5 % 200, 90, 160));
    cv::rectangle(m, {size.width / 4, size.height / 4}, {size.width / 2, size.height / 2}, {255, 255, 255}, cv::FILLED);
    return m;
}

SourceFormat format(cv::Size size = {320, 240}, double fps = 10.0, bool live = false)
{
    return {size, fps, live};
}

EncodeStats write_clip(Encoder &enc, const std::vector<int> &ids, cv::Size size, std::chrono::milliseconds step = 100ms)
{
    const Clock::time_point t0 = Clock::now();
    for (const int id : ids)
        enc.submit(frame(size, id), id, t0 + (id - 1) * step);
    return enc.finish();
}

std::vector<int> range(int first, int last)
{
    std::vector<int> v;
    for (int i = first; i <= last; ++i)
        v.push_back(i);
    return v;
}

struct ReadBack
{
    int frames = 0;
    cv::Size size;
    double last_ms = 0.0;
};

ReadBack read_back(const std::string &path)
{
    cv::VideoCapture cap(path);
    ReadBack r;
    cv::Mat m;
    while (cap.read(m))
    {
        ++r.frames;
        r.size = m.size();
        r.last_ms = cap.get(cv::CAP_PROP_POS_MSEC);
    }
    return r;
}
} // namespace

struct RoundTripCase
{
    const char *file;
    const char *codec;
};

class EncoderRoundTrip : public EncoderTest, public ::testing::WithParamInterface<RoundTripCase>
{
};

TEST_P(EncoderRoundTrip, FrameCountSizeAndDuration)
{
    const auto [file, codec] = GetParam();
    const std::string out = path(file);
    Encoder enc(out, codec, format(), kNoDropSlots);
    const EncodeStats s = write_clip(enc, range(1, 20), {320, 240});
    EXPECT_EQ(s.codec, codec);
    EXPECT_EQ(s.frames, 20);
    EXPECT_EQ(s.dropped, 0);

    const ReadBack r = read_back(out);
    EXPECT_EQ(r.frames, 20);
    EXPECT_EQ(r.size, cv::Size(320, 240));
    EXPECT_NEAR(r.last_ms, 1900.0, 50.0); // frame 20 of a 10 fps clip starts at 1.9 s
}

INSTANTIATE_TEST_SUITE_P(Containers, EncoderRoundTrip,
                         ::testing::Values(RoundTripCase{"a.mp4", "libx264"}, RoundTripCase{"a.mkv", "libx265"},
                                           RoundTripCase{"a.webm", "libvpx-vp9"}),
                         [](const auto &info) { return std::string(info.param.file + 2); }); // "mp4", "mkv", "webm"

TEST_F(EncoderTest, FileGapKeepsTimeline)
{
    const std::string out = path("gap.mkv");
    Encoder enc(out, "libx264", format(), kNoDropSlots);
    std::vector<int> ids = range(1, 20);
    ids.erase(ids.begin() + 4);
    write_clip(enc, ids, {320, 240});

    const ReadBack r = read_back(out);
    EXPECT_EQ(r.frames, 19);
    EXPECT_NEAR(r.last_ms, 1900.0, 50.0); // the gap holds the previous frame, so the clip keeps its length
}

TEST_F(EncoderTest, LiveStampsCaptureTime)
{
    const std::string out = path("live.mkv");
    Encoder enc(out, "libx264", format({320, 240}, 30.0, true), kNoDropSlots);
    write_clip(enc, range(1, 30), {320, 240}, 37ms);

    const ReadBack r = read_back(out);
    EXPECT_EQ(r.frames, 30);
    EXPECT_NEAR(r.last_ms, 29 * 37.0, 34.0); // capture time; the frame ids would end at 967 ms
}

TEST_F(EncoderTest, OddSizeRoundsDownToEven)
{
    const std::string out = path("odd.mp4");
    Encoder enc(out, "libx264", format({321, 241}), kNoDropSlots);
    write_clip(enc, range(1, 5), {321, 241});
    EXPECT_EQ(read_back(out).size, cv::Size(320, 240));
}

TEST_F(EncoderTest, FrameOfAnotherSizeIsScaled)
{
    const std::string out = path("scaled.mp4");
    Encoder enc(out, "libx264", format({320, 240}), kNoDropSlots);
    write_clip(enc, range(1, 5), {640, 480});
    const ReadBack r = read_back(out);
    EXPECT_EQ(r.frames, 5);
    EXPECT_EQ(r.size, cv::Size(320, 240));
}

TEST_F(EncoderTest, BadTargetFailsInConstructor)
{
    EXPECT_THROW(Encoder(path("a.webm"), "libx264", format()), std::runtime_error); // webm cannot hold H.264
    EXPECT_THROW(Encoder("/nonexistent/dir/x.mp4", "libx264", format()), std::runtime_error);
    EXPECT_THROW(Encoder(path("x.zzz"), "libx264", format()), std::runtime_error);
    EXPECT_THROW(Encoder(path("a.ts"), "libvpx-vp9", format()), std::runtime_error); // mpegts cannot carry VP9
    EXPECT_THROW(Encoder("udp://127.0.0.1:5999", "libvpx-vp9", format()), std::runtime_error);
    EXPECT_THROW(Encoder(path("a.mp4"), "nope", format()), std::invalid_argument);
    EXPECT_THROW(Encoder(path("a.mp4"), "libx264", format({0, 0})), std::invalid_argument);
}

TEST_F(EncoderTest, FinishIsIdempotent)
{
    Encoder enc(path("twice.mp4"), "libx264", format(), kNoDropSlots);
    const EncodeStats first = write_clip(enc, range(1, 3), {320, 240});
    const EncodeStats second = enc.finish();
    EXPECT_EQ(second.frames, first.frames);
    EXPECT_EQ(second.dropped, first.dropped);
}

TEST_F(EncoderTest, DropsInsteadOfBlocking)
{
    cv::Mat big(1080, 1920, CV_8UC3);
    cv::randu(big, 0, 256); // noise, so each 1080p frame is slow to encode
    Encoder enc(path("slow.mkv"), "libx265", format({1920, 1080}, 30.0), 1);
    const Clock::time_point t0 = Clock::now();
    for (int id = 1; id <= 30; ++id)
        enc.submit(big, id, t0);
    const EncodeStats s = enc.finish();
    EXPECT_EQ(s.frames + s.dropped, 30);
    EXPECT_GT(s.dropped, 0);
}

TEST_F(EncoderTest, ThreadErrorSurfacesFromSubmit)
{
    const std::string out = path("full.ts");
    fs::create_symlink("/dev/full", out); // every write fails with ENOSPC
    Encoder enc(out, "libx264", format({320, 240}, 30.0), kNoDropSlots);

    cv::Mat noise(240, 320, CV_8UC3);
    cv::randu(noise, 0, 256);
    bool threw = false;
    for (int id = 1; id <= 300 && !threw; ++id)
    {
        try
        {
            enc.submit(noise, id, Clock::now());
        }
        catch (const std::runtime_error &e)
        {
            threw = true;
            EXPECT_NE(std::string(e.what()).find("No space left"), std::string::npos) << e.what();
        }
        std::this_thread::sleep_for(5ms);
    }
    EXPECT_TRUE(threw);
    EXPECT_THROW((void)enc.finish(), std::runtime_error);
}
