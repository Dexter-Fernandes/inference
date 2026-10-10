#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <set>
#include <sstream>
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

long rss_kb()
{
    std::ifstream f("/proc/self/status");
    for (std::string k; f >> k;)
        if (k == "VmRSS:")
        {
            long v;
            f >> v;
            return v;
        }
    return -1;
}

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
        std::ostringstream label;
        label << name << " " << cv::format("%.2f", d.conf);

        int base = 0;
        const cv::Size ts = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        const int y = std::max(d.box.y, ts.height + 4);
        cv::rectangle(frame, {d.box.x, y - ts.height - 4}, {d.box.x + ts.width + 4, y}, color, cv::FILLED);
        cv::putText(frame, label.str(), {d.box.x + 2, y - 3},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1, cv::LINE_AA);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cout << "Usage: " << argv[0]
                  << " <video path|camera_index> <model.onnx>\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string source = argc > 2 ? argv[2] : "0";

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

    // Pace playback with the file's FPS
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0)
        fps = 30.0;
    const int delay_ms = std::max(1, static_cast<int>(1000.0 / fps));

    cv::Mat frame, input, resized, blob;
    std::vector<cv::Mat> outputs;
    LetterboxInfo lb{};
    Config config;
    std::vector<std::string> names; // empty is fine: draw() falls back to class ids
    {
        std::ifstream f("assets/coco.names");
        if (!f)
            std::cerr << "Warning: assets/coco.names not found, showing class ids\n";
        for (std::string line; std::getline(f, line);)
            names.push_back(line);
    }

    const auto out_names = net.getUnconnectedOutLayersNames();
    while (cap.read(frame))
    {
        if (is_camera)
            cv::flip(frame, frame, 1);

        letterbox(frame, resized, input, config.input_size, lb);
        cv::dnn::blobFromImage(input, blob, 1.0 / 255.0, {config.input_size, config.input_size}, cv::Scalar(), /*swapRB=*/true, /*crop=*/false);

        net.setInput(blob);
        net.forward(outputs, out_names);

        const auto dets = postprocess(outputs[0], lb, frame.size(), config);
        draw(frame, dets, names);

        cv::imshow("Playback", frame);

        static int n = 0;
        if (++n % 100 == 0)
            std::cerr << "frame " << n << " rss " << rss_kb() / 1024 << " MB\n";

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27)
            break;
    }
}