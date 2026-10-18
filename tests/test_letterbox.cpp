#include "preprocess.hpp"

#include <gtest/gtest.h>

#include <ostream>
#include <string>

namespace
{
struct Case
{
    const char *name;
    cv::Size src;
    int size;
    cv::Size resized;
    float scale;
    int pad_x, pad_y;
};

void PrintTo(const Case &c, std::ostream *os) { *os << c.name; }

class LetterboxInfo_ : public testing::TestWithParam<Case>
{
};

TEST_P(LetterboxInfo_, SizeAndPadding)
{
    const Case &c = GetParam();
    LetterboxInfo lb{};
    const cv::Size rs = letterbox_info(c.src, c.size, lb);
    EXPECT_EQ(rs, c.resized);
    EXPECT_FLOAT_EQ(lb.scale, c.scale);
    EXPECT_EQ(lb.pad_x, c.pad_x);
    EXPECT_EQ(lb.pad_y, c.pad_y);
}

INSTANTIATE_TEST_SUITE_P(
    Geometries, LetterboxInfo_,
    testing::Values(
        Case{"landscape", {1920, 1080}, 640, {640, 360}, 1.f / 3.f, 0, 140},
        Case{"portrait", {1080, 1920}, 640, {360, 640}, 1.f / 3.f, 140, 0},
        Case{"square", {500, 500}, 640, {640, 640}, 1.28f, 0, 0},
        Case{"upscale", {320, 180}, 640, {640, 360}, 2.f, 0, 140},
        // 337 * 0.64 = 215.68 rounds to 216 (truncating would give 215); the 424 px of padding splits 212/212
        Case{"rounds_up", {1000, 337}, 640, {640, 216}, 0.64f, 0, 212},
        // 333 * 0.64 = 213.12; 427 px of padding is odd, so the extra pixel goes below (pad_y floors)
        Case{"odd_padding", {1000, 333}, 640, {640, 213}, 0.64f, 0, 213},
        Case{"other_size", {1920, 1080}, 320, {320, 180}, 1.f / 6.f, 0, 70}),
    [](const testing::TestParamInfo<Case> &i) { return std::string(i.param.name); });
} // namespace
