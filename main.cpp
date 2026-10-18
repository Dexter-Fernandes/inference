#include <nvtx3/nvtx3.hpp>
#include <opencv2/opencv.hpp>
#include <CLI/CLI.hpp>
#include <cuda_runtime.h>
#include <npp.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

// Fills lb and returns the resized (unpadded) size.
cv::Size letterbox_info(cv::Size src, int size, LetterboxInfo &lb)
{
    lb.scale = std::min(size / static_cast<float>(src.width),
                        size / static_cast<float>(src.height));
    const int nw = static_cast<int>(std::round(src.width * lb.scale));
    const int nh = static_cast<int>(std::round(src.height * lb.scale));
    lb.pad_x = (size - nw) / 2;
    lb.pad_y = (size - nh) / 2;
    return {nw, nh};
}

void letterbox(const cv::Mat &src, cv::Mat &resized, cv::Mat &out, int size, LetterboxInfo &lb)
{
    const cv::Size rs = letterbox_info(src.size(), size, lb);
    const int nw = rs.width, nh = rs.height;

    cv::resize(src, resized, rs, 0, 0, cv::INTER_LINEAR);

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

void check(NppStatus s)
{
    if (s < NPP_SUCCESS)
        throw std::runtime_error("NPP error " + std::to_string(s));
}

void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));
}

struct NppPre
{
    NppStreamContext ctx{};
    cv::Size src_size;
    int size = 0;
    Npp8u *d_src = nullptr, *d_resized = nullptr, *d_box = nullptr;
    Npp32f *d_f32 = nullptr, *d_blob = nullptr;

    NppPre() { check(nppGetStreamContext(&ctx)); }
    ~NppPre() { release(); }
    NppPre(const NppPre &) = delete;
    NppPre &operator=(const NppPre &) = delete;

    void release()
    {
        for (void *p : {(void *)d_src, (void *)d_resized, (void *)d_box, (void *)d_f32, (void *)d_blob})
            cudaFree(p);
        d_src = d_resized = d_box = nullptr;
        d_f32 = d_blob = nullptr;
    }

    // No-op unless the frame or input size changes, so in practice runs once.
    void ensure(cv::Size src, int sz)
    {
        if (src == src_size && sz == size)
            return;
        release();
        src_size = src;
        size = sz;
        const size_t src_bytes = static_cast<size_t>(src.area()) * 3;
        const size_t box_px = static_cast<size_t>(sz) * sz;
        check(cudaMalloc(&d_src, src_bytes));
        check(cudaMalloc(&d_resized, box_px * 3)); // resized fits inside the box
        check(cudaMalloc(&d_box, box_px * 3));
        check(cudaMalloc(&d_f32, box_px * 3 * sizeof(Npp32f)));
        check(cudaMalloc(&d_blob, box_px * 3 * sizeof(Npp32f)));
    }
};

void preprocess_npp(FrameData &framedata, int size, NppPre &npp)
{
    const cv::Mat &src = framedata.frame;
    LetterboxInfo &lb = framedata.lb;
    const cv::Size rs = letterbox_info(src.size(), size, lb);
    const int nw = rs.width, nh = rs.height;

    npp.ensure(src.size(), size);
    const int src_step = src.cols * 3;
    const int rs_step = nw * 3;
    const int box_step = size * 3;
    const int f32_step = size * 3 * sizeof(Npp32f);
    const NppiSize box_sz{size, size};

    {
        nvtx3::scoped_range r{"h2d"};
        check(cudaMemcpy2D(npp.d_src, src_step, src.data, src.step, src_step, src.rows, cudaMemcpyHostToDevice));
    }

    {
        nvtx3::scoped_range r{"npp"};
        // Not nppiResize: its LINEAR uses a different pixel-centre mapping and degrades to nearest
        // on integer downscales. SqrPixel with zero shift matches cv::resize INTER_LINEAR.
        check(nppiResizeSqrPixel_8u_C3R_Ctx(npp.d_src, {src.cols, src.rows}, src_step, {0, 0, src.cols, src.rows},
                                            npp.d_resized, rs_step, {0, 0, nw, nh},
                                            nw / static_cast<double>(src.cols), nh / static_cast<double>(src.rows),
                                            0.0, 0.0, NPPI_INTER_LINEAR, npp.ctx));

        const Npp8u pad[3] = {114, 114, 114};
        check(nppiCopyConstBorder_8u_C3R_Ctx(npp.d_resized, rs_step, {nw, nh},
                                             npp.d_box, box_step, box_sz, lb.pad_y, lb.pad_x, pad, npp.ctx));

        const int bgr2rgb[3] = {2, 1, 0};
        check(nppiSwapChannels_8u_C3IR_Ctx(npp.d_box, box_step, box_sz, bgr2rgb, npp.ctx));

        check(nppiConvert_8u32f_C3R_Ctx(npp.d_box, box_step, npp.d_f32, f32_step, box_sz, npp.ctx));

        const Npp32f inv255[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
        check(nppiMulC_32f_C3IR_Ctx(inv255, npp.d_f32, f32_step, box_sz, npp.ctx));

        // HWC -> NCHW
        const size_t plane = static_cast<size_t>(size) * size;
        Npp32f *const planes[3] = {npp.d_blob, npp.d_blob + plane, npp.d_blob + 2 * plane};
        check(nppiCopy_32f_C3P3R_Ctx(npp.d_f32, f32_step, planes, size * sizeof(Npp32f), box_sz, npp.ctx));
    }

    {
        nvtx3::scoped_range r{"d2h"};
        const int dims[4] = {1, 3, size, size};
        framedata.blob.create(4, dims, CV_32F);
        // Blocking copy on the default stream, so it also waits for the NPP work.
        check(cudaMemcpy(framedata.blob.data, npp.d_blob, 3 * static_cast<size_t>(size) * size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    }
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

    bool use_npp = true;
    app.add_flag("--npp", use_npp, "Preprocess on GPU with NPP");

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
    NppPre npp;

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
            if (use_npp)
                preprocess_npp(fd, config.input_size, npp);
            else
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