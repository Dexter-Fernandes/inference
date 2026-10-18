#include <nvtx3/nvtx3.hpp>
#include <opencv2/opencv.hpp>
#include <CLI/CLI.hpp>
#include <cuda_runtime.h>
#include <npp.h>
#include <NvInfer.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
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
    cv::Mat out;      // net output rows that passed the GPU objectness filter
    int num_kept = 0; // valid rows at the start of out
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

// filter.cu
void filter_rows(const float *d_out, int num_boxes, int stride, float thr, float *d_kept, int *d_count, cudaStream_t s);

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

struct TrtLogger : nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
};

size_t volume(const nvinfer1::Dims &d)
{
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i)
        v *= d.d[i];
    return v;
}

// Owns the engine, one input and one output device buffer, and the stream everything runs on.
struct Trt
{
    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    float *d_in = nullptr, *d_out = nullptr;
    float *d_kept = nullptr; // filtered output rows, same capacity as d_out
    int *d_count = nullptr;
    nvinfer1::Dims in_dims{}, out_dims{};

    explicit Trt(const std::string &engine_path)
    {
        std::ifstream f(engine_path, std::ios::binary);
        if (!f)
            throw std::runtime_error("cannot open engine " + engine_path);
        const std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        runtime.reset(nvinfer1::createInferRuntime(logger));
        engine.reset(runtime->deserializeCudaEngine(blob.data(), blob.size()));
        if (!engine)
            throw std::runtime_error("failed to deserialize engine " + engine_path);
        context.reset(engine->createExecutionContext());
        check(cudaStreamCreate(&stream));

        if (engine->getNbIOTensors() != 2)
            throw std::runtime_error("expected 1 input and 1 output tensor");
        for (int i = 0; i < engine->getNbIOTensors(); ++i)
        {
            const char *name = engine->getIOTensorName(i);
            if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
                throw std::runtime_error(std::string("tensor ") + name + " is not float32");
            const bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            const nvinfer1::Dims d = engine->getTensorShape(name);
            float *&buf = is_input ? d_in : d_out;
            (is_input ? in_dims : out_dims) = d;
            check(cudaMalloc(&buf, volume(d) * sizeof(float)));
            context->setTensorAddress(name, buf);
        }
        check(cudaMalloc(&d_kept, volume(out_dims) * sizeof(float)));
        check(cudaMalloc(&d_count, sizeof(int)));
    }

    ~Trt()
    {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_kept);
        cudaFree(d_count);
        cudaStreamDestroy(stream);
    }
    Trt(const Trt &) = delete;
    Trt &operator=(const Trt &) = delete;
};

struct NppPre
{
    NppStreamContext ctx{};
    int size = 0;
    cv::Mat resized; // host, reused across frames
    Npp8u *d_resized = nullptr, *d_box = nullptr;
    Npp32f *d_f32 = nullptr;

    explicit NppPre(cudaStream_t stream)
    {
        check(nppGetStreamContext(&ctx));
        ctx.hStream = stream;
        check(cudaStreamGetFlags(stream, &ctx.nStreamFlags));
    }
    ~NppPre() { release(); }
    NppPre(const NppPre &) = delete;
    NppPre &operator=(const NppPre &) = delete;

    void release()
    {
        for (void *p : {(void *)d_resized, (void *)d_box, (void *)d_f32})
            cudaFree(p);
        d_resized = d_box = nullptr;
        d_f32 = nullptr;
    }

    // No-op unless the input size changes, so in practice runs once.
    void ensure(int sz)
    {
        if (sz == size)
            return;
        release();
        size = sz;
        const size_t box_px = static_cast<size_t>(sz) * sz;
        check(cudaMalloc(&d_resized, box_px * 3)); // resized fits inside the box
        check(cudaMalloc(&d_box, box_px * 3));
        check(cudaMalloc(&d_f32, box_px * 3 * sizeof(Npp32f)));
    }
};

// Resize on the CPU, upload the small image, then pad/convert/normalise on the GPU
// straight into the engine's input buffer.
void preprocess_npp(FrameData &framedata, int size, NppPre &npp, Trt &trt)
{
    LetterboxInfo &lb = framedata.lb;
    const cv::Size rs = letterbox_info(framedata.frame.size(), size, lb);
    const int nw = rs.width, nh = rs.height;

    npp.ensure(size);
    const int rs_step = nw * 3;
    const int box_step = size * 3;
    const int f32_step = size * 3 * sizeof(Npp32f);
    const NppiSize box_sz{size, size};

    cv::resize(framedata.frame, npp.resized, rs, 0, 0, cv::INTER_LINEAR);

    {
        nvtx3::scoped_range r{"h2d"};
        check(cudaMemcpy2DAsync(npp.d_resized, rs_step, npp.resized.data, npp.resized.step, rs_step, nh,
                                cudaMemcpyHostToDevice, trt.stream));
    }

    {
        nvtx3::scoped_range r{"npp"};
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
        Npp32f *const planes[3] = {trt.d_in, trt.d_in + plane, trt.d_in + 2 * plane};
        check(nppiCopy_32f_C3P3R_Ctx(npp.d_f32, f32_step, planes, size * sizeof(Npp32f), box_sz, npp.ctx));

        check(cudaStreamSynchronize(trt.stream)); // so t.pre includes the GPU work
    }
}

void preprocess_cpu(FrameData &framedata, int size, Trt &trt)
{
    preprocess(framedata, size);
    check(cudaMemcpyAsync(trt.d_in, framedata.blob.data, framedata.blob.total() * sizeof(float),
                          cudaMemcpyHostToDevice, trt.stream));
    check(cudaStreamSynchronize(trt.stream));
}

// Runs the engine, then drops rows below the objectness threshold on the GPU so only
// the survivors cross PCIe (a few hundred rows instead of the full 8.6 MB output).
void infer(Trt &trt, FrameData &framedata, const Config &cfg)
{
    const int num_boxes = trt.out_dims.d[1];
    const int stride = trt.out_dims.d[2];
    {
        nvtx3::scoped_range r{"trt"};
        if (!trt.context->enqueueV3(trt.stream))
            throw std::runtime_error("enqueueV3 failed");
    }
    {
        nvtx3::scoped_range r{"filter"};
        filter_rows(trt.d_out, num_boxes, stride, cfg.conf_thr, trt.d_kept, trt.d_count, trt.stream);
        check(cudaGetLastError());
    }
    {
        nvtx3::scoped_range r{"d2h"};
        check(cudaMemcpyAsync(&framedata.num_kept, trt.d_count, sizeof(int), cudaMemcpyDeviceToHost, trt.stream));
        check(cudaStreamSynchronize(trt.stream));
        check(cudaMemcpyAsync(framedata.out.data, trt.d_kept, static_cast<size_t>(framedata.num_kept) * stride * sizeof(float),
                              cudaMemcpyDeviceToHost, trt.stream));
        check(cudaStreamSynchronize(trt.stream));
    }
}

std::vector<Detection> postprocess(const cv::Mat &out, int num_boxes, const LetterboxInfo &lb, const cv::Size &orig, const Config &cfg)
{
    // YOLOV5 output: [1, num_boxes, 5 + num_classes]
    // row = cx, cy, w, h,  objectness, class scores ...
    // Only the first num_boxes rows are valid.
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
    framedata.dets = postprocess(framedata.out, framedata.num_kept, framedata.lb, framedata.frame.size(), config);
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

    std::string model_path = "assets/yolov5s.engine";
    app.add_option("-m,--model_path", model_path, "Path to the TensorRT engine");

    std::string source = "0";
    app.add_option("-s,--source", source, "Source");

    bool use_npp = true;
    app.add_flag("--npp,!--no-npp", use_npp, "Preprocess on GPU with NPP");

    CLI11_PARSE(app, argc, argv);

    std::unique_ptr<Trt> trt;
    try
    {
        trt = std::make_unique<Trt>(model_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load engine " << model_path << ": " << e.what() << "\n"
                  << "Build one with: trtexec --onnx=assets/yolov5s.onnx --saveEngine=" << model_path << "\n";
        return 1;
    }

    Config config;
    config.input_size = trt->in_dims.d[2]; // NCHW, square

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
    fd.out.create(trt->out_dims.nbDims, std::vector<int>(trt->out_dims.d, trt->out_dims.d + trt->out_dims.nbDims).data(), CV_32F);
    NppPre npp(trt->stream);

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
                preprocess_npp(fd, config.input_size, npp, *trt);
            else
                preprocess_cpu(fd, config.input_size, *trt);
            t.pre.stop();
        }

        {
            nvtx3::scoped_range r{"infer"};
            t.infer.start();
            infer(*trt, fd, config);
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