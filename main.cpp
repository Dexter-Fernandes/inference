#include <opencv2/opencv.hpp>

// #include <algorithm>
// #include <cctype>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    const std::string source = argc > 1 ? argv[1] : "0";

    cv::VideoCapture cap;
    // const bool is_camera = std::ranges::all_of(
    //     source, [](unsigned char c)
    //     { return std::isdigit(c); });
    const bool is_camera = source.find_first_not_of("0123456789") == std::string::npos;
    // is_camera ? cap.open(std::stoi(source)) : cap.open(source);
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

    // cv::namedWindow("Playback", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    while (cap.read(frame))
    {
        if (is_camera)
            cv::flip(frame, frame, 1);

        cv::imshow("Playback", frame);
        const int key = cv::waitKey(delay_ms) & 0xFF;
        if (key == 'q' || key == 27)
            break;
    }

    // return 0;
}