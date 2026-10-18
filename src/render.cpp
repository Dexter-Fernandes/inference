#include "render.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

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

void render(Frame &frame, const std::vector<std::string> &names, double fps)
{
    draw(frame.image, frame.dets, names);

    const std::string hud = cv::format("FPS: %.1f  Det: %zu", fps, frame.dets.size());
    cv::putText(frame.image, hud, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 4, cv::LINE_AA);
    cv::putText(frame.image, hud, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 1, cv::LINE_AA);
}
