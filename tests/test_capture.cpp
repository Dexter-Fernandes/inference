#include "capture.hpp"

#include <gtest/gtest.h>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <stdexcept>

TEST(IsCameraSource, DigitsOnly)
{
    EXPECT_TRUE(is_camera_source("0"));
    EXPECT_TRUE(is_camera_source("12"));
    EXPECT_FALSE(is_camera_source(""));
    EXPECT_FALSE(is_camera_source("0.mp4"));
    EXPECT_FALSE(is_camera_source("-1"));
    EXPECT_FALSE(is_camera_source("rtsp://host/1"));
}

TEST(Capture, MissingFileThrows)
{
    EXPECT_THROW(Capture("/nonexistent/video.mp4"), std::runtime_error);
}

TEST(Capture, FileEndsAtEofWithoutReconnects)
{
    const auto path = (std::filesystem::temp_directory_path() / "capture_test.avi").string();
    {
        cv::VideoWriter w(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, {64, 48});
        ASSERT_TRUE(w.isOpened());
        for (int i = 0; i < 3; i++)
            w.write(cv::Mat(48, 64, CV_8UC3, cv::Scalar(i * 40, 0, 0)));
    }
    Capture cap(path);
    cv::Mat frame;
    for (int i = 0; i < 3; i++)
        EXPECT_TRUE(cap.read(frame)) << i;
    EXPECT_EQ(frame.size(), cv::Size(64, 48));
    EXPECT_FALSE(cap.read(frame));
    EXPECT_EQ(cap.reconnects(), 0);
    std::filesystem::remove(path);
}
