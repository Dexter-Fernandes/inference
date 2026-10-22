#include "render.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

const std::array<cv::Scalar, kPaletteSize> &palette()
{
    static const std::array<cv::Scalar, kPaletteSize> p = {{
        {60, 76, 231},  {113, 204, 46}, {219, 152, 52}, {15, 196, 241}, {182, 89, 155}, {156, 188, 26},
        {34, 126, 230}, {173, 68, 142}, {122, 160, 255}, {96, 174, 39}, {200, 200, 60},  {128, 128, 255},
    }};
    return p;
}

void draw_boxes(cv::Mat &frame, const std::vector<Detection> &dets,
                const std::function<std::optional<cv::Scalar>(int class_id)> &color_of)
{
    for (const auto &d : dets)
        if (const auto color = color_of(d.class_id))
            cv::rectangle(frame, d.box, *color, 2);
}

void render(Frame &frame, const std::vector<std::string> &names)
{
    const auto color = [](int class_id) { return palette()[class_id % kPaletteSize]; };
    draw_boxes(frame.image, frame.dets, [&](int class_id) { return std::optional(color(class_id)); });

    for (const auto &d : frame.dets)
    {
        const std::string name = d.class_id < static_cast<int>(names.size())
                                     ? names[d.class_id]
                                     : std::to_string(d.class_id);
        const std::string label = name + cv::format(" %.2f", d.conf);

        const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
        const int y = std::max(d.box.y, ts.height + 4);
        cv::rectangle(frame.image, {d.box.x, y - ts.height - 4}, {d.box.x + ts.width + 4, y}, color(d.class_id), cv::FILLED);
        cv::putText(frame.image, label, {d.box.x + 2, y - 3},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1, cv::LINE_AA);
    }

    const std::string hud = cv::format("FPS: %.1f  Det: %zu", frame.fps, frame.dets.size());
    cv::putText(frame.image, hud, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 4, cv::LINE_AA);
    cv::putText(frame.image, hud, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 1, cv::LINE_AA);
}
