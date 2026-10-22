#pragma once

#include "common.hpp"
#include "legend.hpp"

#include <opencv2/core/utility.hpp>

#include <array>
#include <string>
#include <vector>

constexpr int kPersonClass = 0; // COCO, the one class whose boxes start switched on

struct GLFWwindow;
struct ImFont;

// The playback window: the video on the left and a hardware-panel sidebar on the right with an FPS/Det
// readout, a legend of the detected classes (each with a switch for its boxes) and a two-row debug screen
// behind a roll-up shutter.
// Everything runs on the thread that calls show(), which must be the one that constructed it.
class Gui
{
public:
    Gui(std::vector<std::string> names, double stream_fps); // stream_fps: the source's nominal rate. Throws when there is no display or GL context
    ~Gui();
    Gui(const Gui &) = delete;
    Gui &operator=(const Gui &) = delete;

    // Draws f's visible boxes onto f.image, shows it beside the sidebar and records the time spent as
    // f.stage_ms.render. False once the window is closed or q/Esc is pressed.
    bool show(Frame &f);

    // The two rows the debug screen shows once the shutter is up; replaces the previous pair.
    void set_stage_rows(std::array<std::string, 2> rows) { stage_rows_ = std::move(rows); }

private:
    void draw_sidebar(const Frame &f, float x0, float w, float h);
    void upload(const cv::Mat &bgr);

    GLFWwindow *window_ = nullptr;
    const std::vector<std::string> names_;
    const double stream_fps_;
    Legend legend_{{kPersonClass}};
    std::array<std::string, 2> stage_rows_;
    float shutter_ = 0.0f;        // 0 closed, 1 open
    bool shutter_open_ = false;   // where it is heading
    cv::TickMeter render_tm_;
    unsigned tex_ = 0;
    int tex_w_ = 0, tex_h_ = 0;
    ImFont *sans_ = nullptr, *mono_ = nullptr, *lcd_ = nullptr;
};
