#include "capture.hpp"
#include "common.hpp"
#include "engine.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include "render.hpp"

#include <nvtx3/nvtx3.hpp>
#include <opencv2/highgui.hpp>
#include <CLI/CLI.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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

    auto [cap, is_camera] = open_source(source);

    if (!cap.isOpened())
    {
        std::cerr << "Failed to open source: " << source << "\n";
        return 1;
    }

    FrameData fd{};
    fd.out.create(trt->out_dims.nbDims, std::vector<int>(trt->out_dims.d, trt->out_dims.d + trt->out_dims.nbDims).data(), CV_32F);
    NppPre npp(trt->stream.get());

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