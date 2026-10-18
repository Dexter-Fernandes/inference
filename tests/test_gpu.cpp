#include "engine.hpp"
#include "filter.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"

#include <gtest/gtest.h>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
const std::string kAssets = ASSETS_DIR;

// Skips the calling test when an asset is missing.
#define REQUIRE_ASSET(path)                                          \
    do                                                               \
    {                                                                \
        if (!fs::exists(path))                                       \
            GTEST_SKIP() << "missing asset " << std::string(path);   \
    } while (0)

// The engine is slow to deserialize, so tests share one when gpu_tests runs as one process
// (ctest runs each test in its own).
class WithEngine : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        if (fs::exists(path()))
            trt_ = std::make_unique<Trt>(path());
    }
    static void TearDownTestSuite() { trt_.reset(); }
    void SetUp() override
    {
        REQUIRE_ASSET(path());
        ASSERT_NE(trt_, nullptr);
    }
    static std::string path() { return kAssets + "/yolov5s.engine"; }

    static std::unique_ptr<Trt> trt_;
};
std::unique_ptr<Trt> WithEngine::trt_;

cv::Mat noise(cv::Size size, uint64 seed)
{
    cv::Mat m(size, CV_8UC3);
    cv::RNG rng(seed);
    rng.fill(m, cv::RNG::UNIFORM, 0, 256);
    return m;
}

// Runs preprocess_npp and reads the first 3*size*size floats of the engine input back.
std::vector<float> npp_input(Trt &trt, NppPre &npp, const cv::Mat &frame, int size)
{
    FrameData fd{};
    fd.frame = frame;
    preprocess_npp(fd, size, npp, trt);
    std::vector<float> got(3 * static_cast<size_t>(size) * size);
    check(cudaMemcpy(got.data(), trt.d_in.get(), got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return got;
}

TEST_F(WithEngine, NppPreprocessMatchesCpuPreprocess)
{
    struct Case
    {
        const char *name;
        cv::Size frame;
        int size;
    };
    // One NppPre throughout. The sequence repeats a geometry (cached border), changes the aspect
    // ratio at a fixed size (border must be redrawn) and changes the input size (ensure() reallocates).
    const Case cases[] = {
        {"landscape640", {1280, 720}, 640}, {"landscape640_again", {1280, 720}, 640},
        {"portrait640", {720, 1280}, 640},  {"square640", {500, 500}, 640},
        {"wide640", {1000, 200}, 640},      {"tall640", {300, 700}, 640},
        {"landscape320", {1280, 720}, 320}, {"tall320", {300, 700}, 320},
        {"landscape640_back", {1280, 720}, 640},
    };

    NppPre npp(trt_->stream.get());
    uint64 seed = 1;
    float worst = 0;
    for (const Case &c : cases)
    {
        SCOPED_TRACE(c.name);
        const cv::Mat frame = noise(c.frame, seed++);
        const std::vector<float> got = npp_input(*trt_, npp, frame, c.size);

        FrameData cpu{};
        cpu.frame = frame;
        preprocess(cpu, c.size);
        ASSERT_EQ(cpu.blob.total() * cpu.blob.channels(), got.size());
        const float *want = cpu.blob.ptr<float>();

        float max_diff = 0;
        for (size_t i = 0; i < got.size(); ++i)
            max_diff = std::max(max_diff, std::fabs(got[i] - want[i]));
        EXPECT_LT(max_diff, 1e-6f);
        worst = std::max(worst, max_diff);
    }
    RecordProperty("npp_max_abs_diff", std::to_string(worst));
    std::cout << "npp vs cpu preprocess: max abs diff " << worst << "\n";
}

TEST(FilterRows, ReturnsExactlyTheRowsAtOrAboveThreshold)
{
    constexpr int num_boxes = 25200, stride = 85;
    constexpr float thr = 0.25f;

    // Element 0 carries the row index; objectness is thr or higher for every 7th row and below it otherwise.
    std::vector<float> host(static_cast<size_t>(num_boxes) * stride);
    std::set<int> expected;
    for (int i = 0; i < num_boxes; ++i)
    {
        float *row = &host[static_cast<size_t>(i) * stride];
        for (int k = 0; k < stride; ++k)
            row[k] = static_cast<float>(i) + k * 1e-3f;
        row[0] = static_cast<float>(i);
        row[4] = i == 7 ? thr : (i % 7 == 0 ? 0.9f : 0.249f);
        if (row[4] >= thr)
            expected.insert(i);
    }

    CudaStream stream;
    auto d_out = make_device<float>(host.size());
    auto d_kept = make_device<float>(host.size());
    auto d_count = make_device<int>(1);
    check(cudaMemcpy(d_out.get(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(filter_rows(d_out.get(), num_boxes, stride, thr, d_kept.get(), d_count.get(), stream.get()));
    check(cudaStreamSynchronize(stream.get()));

    int count = -1;
    check(cudaMemcpy(&count, d_count.get(), sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(count, static_cast<int>(expected.size()));

    std::vector<float> kept(static_cast<size_t>(count) * stride);
    check(cudaMemcpy(kept.data(), d_kept.get(), kept.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::set<int> got;
    for (int j = 0; j < count; ++j)
    {
        const float *row = &kept[static_cast<size_t>(j) * stride];
        const int i = static_cast<int>(row[0]);
        ASSERT_TRUE(got.insert(i).second) << "row " << i << " returned twice";
        // whole row arrives intact, not just the id
        ASSERT_TRUE(std::equal(row, row + stride, &host[static_cast<size_t>(i) * stride])) << "row " << i << " corrupted";
    }
    EXPECT_EQ(got, expected);

    // nothing passes: count is reset to 0 rather than left at the previous value
    check(filter_rows(d_out.get(), num_boxes, stride, 2.f, d_kept.get(), d_count.get(), stream.get()));
    check(cudaStreamSynchronize(stream.get()));
    check(cudaMemcpy(&count, d_count.get(), sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(count, 0);
}

double iou(const cv::Rect &a, const cv::Rect &b)
{
    const double inter = (a & b).area();
    return inter / (a.area() + b.area() - inter);
}

// For each detection in a, the best same-class match in b with IoU > 0.9 (if any).
struct Matches
{
    int matched = 0;
    std::vector<double> conf_diffs;
};

Matches match(const std::vector<Detection> &a, const std::vector<Detection> &b)
{
    Matches m;
    for (const Detection &da : a)
    {
        const Detection *best = nullptr;
        double best_iou = 0.9;
        for (const Detection &db : b)
            if (db.class_id == da.class_id && iou(da.box, db.box) > best_iou)
                best = &db, best_iou = iou(da.box, db.box);
        if (best)
        {
            ++m.matched;
            m.conf_diffs.push_back(std::fabs(da.conf - best->conf));
        }
    }
    return m;
}

TEST_F(WithEngine, TensorRtAgreesWithOnnxOnCpu)
{
    const std::string onnx = kAssets + "/yolov5s.onnx", video = kAssets + "/test.mp4";
    REQUIRE_ASSET(onnx);
    REQUIRE_ASSET(video);

    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnx);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::VideoCapture cap(video);
    ASSERT_TRUE(cap.isOpened());

    Config cfg;
    cfg.input_size = trt_->in_dims.d[2];
    NppPre npp(trt_->stream.get());

    int total_trt = 0, total_cpu = 0, matched_trt = 0, matched_cpu = 0;
    std::vector<double> conf_diffs;
    for (int n = 0; n < 40; ++n)
    {
        FrameData fd{};
        ASSERT_TRUE(cap.read(fd.frame)) << "frame " << n;
        if (n % 4 != 0)
            continue;

        preprocess_npp(fd, cfg.input_size, npp, *trt_);
        const std::vector<int> out_shape(trt_->out_dims.d, trt_->out_dims.d + trt_->out_dims.nbDims);
        fd.out.create(static_cast<int>(out_shape.size()), out_shape.data(), CV_32F);
        infer(*trt_, fd, cfg);
        const auto dets_trt = postprocess(fd.out, fd.num_kept, fd.lb, fd.frame.size(), cfg);

        FrameData ref{};
        ref.frame = fd.frame;
        preprocess(ref, cfg.input_size);
        net.setInput(ref.blob);
        const cv::Mat out_cpu = net.forward();
        ASSERT_EQ(out_cpu.dims, 3);
        const auto dets_cpu = postprocess(out_cpu, out_cpu.size[1], ref.lb, ref.frame.size(), cfg);

        const Matches a = match(dets_trt, dets_cpu), b = match(dets_cpu, dets_trt);
        total_trt += static_cast<int>(dets_trt.size()), total_cpu += static_cast<int>(dets_cpu.size());
        matched_trt += a.matched, matched_cpu += b.matched;
        conf_diffs.insert(conf_diffs.end(), a.conf_diffs.begin(), a.conf_diffs.end());
    }

    ASSERT_GT(total_trt, 0);
    ASSERT_GT(total_cpu, 0);
    ASSERT_FALSE(conf_diffs.empty());
    std::nth_element(conf_diffs.begin(), conf_diffs.begin() + conf_diffs.size() / 2, conf_diffs.end());
    const double median = conf_diffs[conf_diffs.size() / 2];

    std::cout << "trt vs onnx-cpu: trt " << matched_trt << "/" << total_trt << " matched, cpu "
              << matched_cpu << "/" << total_cpu << " matched, median conf diff " << median << "\n";
    RecordProperty("matched_trt", std::to_string(matched_trt) + "/" + std::to_string(total_trt));
    RecordProperty("matched_cpu", std::to_string(matched_cpu) + "/" + std::to_string(total_cpu));
    RecordProperty("median_conf_diff", std::to_string(median));

    EXPECT_GE(matched_trt, 0.99 * total_trt);
    EXPECT_GE(matched_cpu, 0.99 * total_cpu);
    EXPECT_LT(median, 1e-3);
}
} // namespace
