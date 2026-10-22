#include "seven_segment.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

TEST(SegmentMask, EightLightsEverySegment)
{
    EXPECT_EQ(segment_mask('8'), 0b1111111u);
}

TEST(SegmentMask, DigitsLightTheirKnownSegments)
{
    constexpr unsigned a = 1, b = 2, c = 4, d = 8, e = 16, f = 32, g = 64;
    EXPECT_EQ(segment_mask('0'), a | b | c | d | e | f);
    EXPECT_EQ(segment_mask('1'), b | c);
    EXPECT_EQ(segment_mask('2'), a | b | d | e | g);
    EXPECT_EQ(segment_mask('3'), a | b | c | d | g);
    EXPECT_EQ(segment_mask('4'), b | c | f | g);
    EXPECT_EQ(segment_mask('5'), a | c | d | f | g);
    EXPECT_EQ(segment_mask('6'), a | c | d | e | f | g);
    EXPECT_EQ(segment_mask('7'), a | b | c);
    EXPECT_EQ(segment_mask('9'), a | b | c | d | f | g);
}

TEST(SegmentMask, MinusIsTheMiddleBarAndOthersAreDark)
{
    EXPECT_EQ(segment_mask('-'), 64u);
    EXPECT_EQ(segment_mask(' '), 0u);
    EXPECT_EQ(segment_mask('.'), 0u);
    EXPECT_EQ(segment_mask('x'), 0u);
}

TEST(SegmentShapes, StayInsideTheCellAndHaveArea)
{
    for (const float h : {20.0f, 26.0f, 40.0f})
    {
        const float w = digit_width(h);
        for (const Segment &s : segment_shapes(h))
        {
            float twice_area = 0;
            for (size_t i = 0; i < s.size(); ++i)
            {
                const Pt p = s[i], q = s[(i + 1) % s.size()];
                EXPECT_GE(p.x, 0.0f);
                EXPECT_LE(p.x, w);
                EXPECT_GE(p.y, 0.0f);
                EXPECT_LE(p.y, h);
                twice_area += p.x * q.y - q.x * p.y;
            }
            EXPECT_GT(twice_area, 0.0f) << "clockwise on a y-down screen, and not degenerate";
        }
    }
}

TEST(SegmentShapes, MiddleBarSitsAtHalfHeight)
{
    const auto segs = segment_shapes(30.0f);
    float lo = 1e9f, hi = -1e9f;
    for (const Pt &p : segs[6])
    {
        lo = std::min(lo, p.y);
        hi = std::max(hi, p.y);
    }
    EXPECT_NEAR((lo + hi) / 2, 15.0f, 1e-4f);
}
