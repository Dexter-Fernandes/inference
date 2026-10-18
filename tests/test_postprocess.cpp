#include "postprocess.hpp"
#include "preprocess.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace
{
constexpr int kStride = 85; // cx, cy, w, h, objectness, 80 class scores

// 1280x720 into 640: scale 0.5, pad_y 140, so letterbox <-> original maps are exact in float.
const cv::Size kOrig{1280, 720};

class Postprocess : public testing::Test
{
protected:
    void SetUp() override { ASSERT_EQ(letterbox_info(kOrig, 640, lb).width, 640); }

    // Rows are given in letterbox (network input) coordinates.
    void add(float cx, float cy, float w, float h, float obj, int cls, float cls_score)
    {
        std::vector<float> r(kStride, 0.f);
        r[0] = cx, r[1] = cy, r[2] = w, r[3] = h, r[4] = obj, r[5 + cls] = cls_score;
        rows.push_back(std::move(r));
    }

    // Same, but the box is given as an original-image rect (x, y, w, h).
    void add_orig(int x, int y, int w, int h, float obj, int cls, float cls_score)
    {
        add((x + w / 2.f) * lb.scale + lb.pad_x, (y + h / 2.f) * lb.scale + lb.pad_y, w * lb.scale, h * lb.scale, obj, cls, cls_score);
    }

    std::vector<Detection> run(int num_boxes = -1)
    {
        const int sizes[3] = {1, static_cast<int>(rows.size()), kStride};
        cv::Mat out(3, sizes, CV_32F);
        for (size_t i = 0; i < rows.size(); ++i)
            std::copy(rows[i].begin(), rows[i].end(), out.ptr<float>(0, static_cast<int>(i)));
        return postprocess(out, num_boxes < 0 ? static_cast<int>(rows.size()) : num_boxes, lb, kOrig, cfg);
    }

    LetterboxInfo lb{};
    Config cfg; // conf_thr 0.25, nms_thr 0.5
    std::vector<std::vector<float>> rows;
};

TEST_F(Postprocess, BoxMapsBackToOriginalImage)
{
    add_orig(400, 200, 200, 200, 1.f, 7, 0.9f);
    const auto dets = run();
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].class_id, 7);
    EXPECT_FLOAT_EQ(dets[0].conf, 0.9f);
    EXPECT_EQ(dets[0].box, cv::Rect(400, 200, 200, 200));
}

TEST_F(Postprocess, DropsRowsBelowConfidenceThreshold)
{
    add_orig(100, 100, 100, 100, 0.1f, 0, 0.99f);  // objectness too low
    add_orig(300, 100, 100, 100, 0.9f, 1, 0.2f);   // obj * class = 0.18
    add_orig(500, 100, 100, 100, 0.9f, 2, 0.9f);   // obj * class = 0.81
    const auto dets = run();
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].class_id, 2);
    EXPECT_FLOAT_EQ(dets[0].conf, 0.81f);
}

TEST_F(Postprocess, KeepClassesFilters)
{
    add_orig(100, 100, 100, 100, 1.f, 0, 0.9f);
    add_orig(400, 100, 100, 100, 1.f, 2, 0.9f);

    EXPECT_EQ(run().size(), 2u) << "empty keep_classes keeps everything";

    cfg.keep_classes = {2};
    const auto dets = run();
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].class_id, 2);
    EXPECT_EQ(dets[0].box, cv::Rect(400, 100, 100, 100));
}

TEST_F(Postprocess, OverlappingSameClassKeepsHigherScore)
{
    add_orig(400, 200, 200, 200, 1.f, 3, 0.7f);
    add_orig(410, 200, 200, 200, 1.f, 3, 0.9f);
    const auto dets = run();
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_FLOAT_EQ(dets[0].conf, 0.9f);
    EXPECT_EQ(dets[0].box, cv::Rect(410, 200, 200, 200));
}

TEST_F(Postprocess, OverlappingDifferentClassesBothSurvive)
{
    add_orig(400, 200, 200, 200, 1.f, 3, 0.7f);
    add_orig(410, 200, 200, 200, 1.f, 4, 0.9f);
    const auto dets = run();
    ASSERT_EQ(dets.size(), 2u);
    EXPECT_NE(dets[0].class_id, dets[1].class_id);
}

TEST_F(Postprocess, BoxPartlyOutsideImageIsClipped)
{
    add_orig(-40, 100, 160, 100, 1.f, 0, 0.9f);   // hangs off the left edge
    add_orig(600, 650, 100, 100, 1.f, 1, 0.9f);   // hangs off the bottom edge (y 650..750 of 720)
    const auto dets = run();
    ASSERT_EQ(dets.size(), 2u);
    for (const Detection &d : dets)
    {
        if (d.class_id == 0)
            EXPECT_EQ(d.box, cv::Rect(0, 100, 120, 100));
        else
            EXPECT_EQ(d.box, cv::Rect(600, 650, 100, 70));
    }
}

TEST_F(Postprocess, BoxEntirelyOutsideImageIsDropped)
{
    add_orig(100, -300, 100, 100, 1.f, 0, 0.9f); // above the image
    add_orig(100, 100, 100, 100, 1.f, 1, 0.9f);
    const auto dets = run();
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].class_id, 1);
}

TEST_F(Postprocess, OnlyFirstNumBoxesRowsAreRead)
{
    add_orig(100, 100, 100, 100, 1.f, 0, 0.9f);
    add_orig(400, 100, 100, 100, 1.f, 1, 0.99f); // stale row past the valid count

    const auto first = run(1);
    ASSERT_EQ(first.size(), 1u);
    EXPECT_EQ(first[0].class_id, 0);

    EXPECT_EQ(run(2).size(), 2u) << "same row is returned once it is within num_boxes";
    EXPECT_TRUE(run(0).empty());
}
} // namespace
