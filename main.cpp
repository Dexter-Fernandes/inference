#include <nvtx3/nvtx3.hpp>
#include <opencv2/opencv.hpp>
#include <CLI/CLI.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <set>
#include <vector>

struct Config
{
    int input_size = 640;
    float conf_thr = 0.25f;
    float nms_thr = 0.5f;
    std::set<int> keep_classes; // empty = keep all
};

struct Detection
{
    int class_id;
    float conf;
    cv::Rect box;
};

struct LetterboxInfo
{
    float scale;
    int pad_x, pad_y;
};

struct FrameData
{
    cv::Mat frame; // original, drawn on at the end
    cv::Mat blob;  // NCHW input to the net
    LetterboxInfo lb;
    cv::Mat out; // raw net output
    std::vector<Detection> dets;
};

struct StageTimers
{
    cv::TickMeter read, pre, infer, post, render;
};

void letterbox(const cv::Mat &src, cv::Mat &resized, cv::Mat &out, int size, LetterboxInfo &lb)
{
    lb.scale = std::min(size / static_cast<float>(src.cols),
                        size / static_cast<float>(src.rows));
    const int nw = static_cast<int>(std::round(src.cols * lb.scale));
    const int nh = static_cast<int>(std::round(src.rows * lb.scale));
    lb.pad_x = (size - nw) / 2;
    lb.pad_y = (size - nh) / 2;

    cv::resize(src, resized, {nw, nh}, 0, 0, cv::INTER_LINEAR);

    out.create(size, size, src.type()); // no-op after the first frame if size and type match
    out.setTo(cv::Scalar::all(114));    // memset, no allocation
    resized.copyTo(out(cv::Rect(lb.pad_x, lb.pad_y, nw, nh)));
}

void preprocess(FrameData &framedata, int size)
{
    cv::Mat resized, letterboxed;

    letterbox(framedata.frame, resized, letterboxed, size, framedata.lb);
    cv::dnn::blobFromImage(letterboxed, framedata.blob, 1.0 / 255.0, {size, size}, cv::Scalar(), /*swapRB=*/true, /*crop=*/false);
}

void infer(cv::dnn::Net &net, FrameData &framedata)
{
    net.setInput(framedata.blob);
    framedata.out = net.forward().clone();
}

std::vector<Detection> postprocess(const cv::Mat &out, const LetterboxInfo &lb, const cv::Size &orig, const Config &cfg)
{
    // YOLOV5 output: [1, num_boxes, 5 + num_classes]
    // row = cx, cy, w, h,  objectness, class scores ...
    const int num_boxes = out.size[1];
    const int stride = out.size[2];
    const int num_cls = stride - 5;
    const auto *data = reinterpret_cast<const float *>(out.data);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < num_boxes; ++i)
    {
        const float *row = data + static_cast<size_t>(i) * stride;
        const float obj = row[4];
        if (obj < cfg.conf_thr)
            continue;

        const float *cls_begin = row + 5;
        const float *best = std::max_element(cls_begin, cls_begin + num_cls);
        const int cls = static_cast<int>(best - cls_begin);
        const float conf = obj * (*best);
        if (conf < cfg.conf_thr)
            continue;
        if (!cfg.keep_classes.empty() && !cfg.keep_classes.count(cls))
            continue;

        // Undo letterbox
        const float x1 = (row[0] - row[2] / 2.f - lb.pad_x) / lb.scale;
        const float y1 = (row[1] - row[3] / 2.f - lb.pad_y) / lb.scale;
        const float x2 = (row[0] + row[2] / 2.f - lb.pad_x) / lb.scale;
        const float y2 = (row[1] + row[3] / 2.f - lb.pad_y) / lb.scale;

        cv::Rect box(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                     cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        box &= cv::Rect(0, 0, orig.width, orig.height);
        if (box.area() <= 0)
            continue;

        boxes.push_back(box);
        scores.push_back(conf);
        class_ids.push_back(cls);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxesBatched(boxes, scores, class_ids,
                             cfg.conf_thr, cfg.nms_thr, keep);

    std::vector<Detection> dets;
    dets.reserve(keep.size());
    for (int idx : keep)
        dets.push_back({class_ids[idx], scores[idx], boxes[idx]});
    return dets;
}

void postprocess_stage(FrameData &framedata, const Config &config)
{
    framedata.dets = postprocess(framedata.out, framedata.lb, framedata.frame.size(), config);
}

void draw(cv::Mat &frame, const std::vector<Detection> &dets,
          const std::vector<std::string> &names)
{
    for (const auto &d : dets)
    {
        const cv::Scalar color(
            (37 * d.class_id) % 256, (91 * d.class_id) % 256, (151 * d.class_id) % 256);
        cv::rectangle(frame, d.box, color, 2);

        const std::string name = d.class_id < static_cast<int>(names.size())
                                     ? names[d.class_id]
                                     : std::to_string(d.class_id);
        const std::string label = name + cv::format(" %.2f", d.conf);

        const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
        const int y = std::max(d.box.y, ts.height + 4);
        cv::rectangle(frame, {d.box.x, y - ts.height - 4}, {d.box.x + ts.width + 4, y}, color, cv::FILLED);
        cv::putText(frame, label, {d.box.x + 2, y - 3},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1, cv::LINE_AA);
    }
}

void render(FrameData &framedata, const std::vector<std::string> &names, double fps)
{
    draw(framedata.frame, framedata.dets, names);

    const std::string hud = cv::format("FPS: %.1f  Det: %zu", fps, framedata.dets.size());
    cv::putText(framedata.frame, hud, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 4, cv::LINE_AA);
    cv::putText(framedata.frame, hud, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 1, cv::LINE_AA);

    cv::imshow("Playback", framedata.frame);
}

bool read_frame(cv::VideoCapture &cap, bool flip, FrameData &framedata)
{
    if (!cap.read(framedata.frame))
        return false;
    if (flip)
        cv::flip(framedata.frame, framedata.frame, 1);
    return true;
}

int main(int argc, char **argv)
{

    CLI::App app("Runs an object detection on a video or webcam");
    argv = app.ensure_utf8(argv);

    std::string model_path = "assets/yolov5s.onnx";
    app.add_option("-m,--model_path", model_path, "Path to the model.onnx");

    std::string source = "0";
    app.add_option("-s,--source", source, "Source");

    Config config;
    app.add_option("-i,--input_size", config.input_size, "Model input size (must match the exported model)");

    CLI11_PARSE(app, argc, argv);

    cv::dnn::Net net;
    try
    {
        net = cv::dnn::readNetFromONNX(model_path);
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "Failed to load model " << model_path << ":\n"
                  << e.what() << "\n";
        return 1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::VideoCapture cap;
    // is_camera if source only contains numbers
    const bool is_camera = !source.empty() && source.find_first_not_of("0123456789") == std::string::npos;
    if (is_camera)
        cap.open(std::stoi(source));
    else
        cap.open(source);

    if (!cap.isOpened())
    {
        std::cerr << "Failed to open source: " << source << "\n";
        return 1;
    }

    FrameData fd{};

    std::vector<std::string> names; // empty is fine: draw() falls back to class ids
    {
        std::ifstream f("assets/coco.names");
        if (!f)
            std::cerr << "Warning: assets/coco.names not found, showing class ids\n";
        for (std::string line; std::getline(f, line);)
            names.push_back(line);
    }

    cv::TickMeter frame_tm;
    StageTimers t{};
    double fps_ema = 0.0;
    for (int n = 1;; n++)
    {
        if (n == 10)
            for (auto *tm : {&t.read, &t.pre, &t.infer, &t.post, &t.render})
                tm->reset();

        bool ret;
        {
            nvtx3::scoped_range r{"read"};
            t.read.start();
            ret = read_frame(cap, is_camera, fd);
            t.read.stop();
        }
        if (!ret)
        {
            break;
        }

        frame_tm.reset();
        frame_tm.start();

        {
            nvtx3::scoped_range r{"pre"};
            t.pre.start();
            preprocess(fd, config.input_size);
            t.pre.stop();
        }

        {
            nvtx3::scoped_range r{"infer"};
            t.infer.start();
            infer(net, fd);
            t.infer.stop();
        }

        {
            nvtx3::scoped_range r{"post"};
            t.post.start();
            postprocess_stage(fd, config);
            t.post.stop();
        }

        frame_tm.stop();
        const double fps = 1.0 / frame_tm.getTimeSec();
        if (fps_ema == 0.0)
            fps_ema = fps;
        else
            fps_ema = 0.9 * fps_ema + 0.1 * fps;

        {
            nvtx3::scoped_range r{"render"};
            t.render.start();
            render(fd, names, fps_ema);
            t.render.stop();
        }

        if (n % 100 == 0)
        {
            std::cerr << cv::format("frame %d: read %.1f | pre %.1f | infer %.1f | post %.1f | render %.1f ms\n",
                                    n, t.read.getAvgTimeMilli(), t.pre.getAvgTimeMilli(), t.infer.getAvgTimeMilli(), t.post.getAvgTimeMilli(), t.render.getAvgTimeMilli());
        }

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27)
            break;
    }
}