#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <set>

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

    // cv::Mat resized;
    cv::resize(src, resized, {nw, nh}, 0, 0, cv::INTER_LINEAR);

    // cv::Mat out(size, size, src.type(), cv::Scalar::all(114));
    out.create(size, size, src.type()); // no-op after the first frame if size and type match
    out.setTo(cv::Scalar::all(114));    // memset, no allocation
    resized.copyTo(out(cv::Rect(lb.pad_x, lb.pad_y, nw, nh)));

    // return out;
}

int main(int argc, char **argv)
{
    const std::string source = argc > 1 ? argv[1] : "0";

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

    cv::Mat frame, out_frame, resized;
    LetterboxInfo lb{};
    Config config;
    while (cap.read(frame))
    {
        if (is_camera)
            cv::flip(frame, frame, 1);

        letterbox(frame, resized, out_frame, config.input_size, lb);

        cv::imshow("Playback", out_frame);

        static int n = 0;
        if (++n % 100 == 0)
            std::cerr << "frame " << n << " rss " << rss_kb() / 1024 << " MB\n";

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27)
            break;
    }

    // return 0;
}