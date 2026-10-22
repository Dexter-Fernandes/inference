#include "gui.hpp"
#include "render.hpp"
#include "timing.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#ifndef GL_BGR
#define GL_BGR 0x80E0
#endif

namespace
{
constexpr float kSidebarW = 320.0f;
constexpr float kSansPx = 15.0f, kMonoPx = 11.0f, kLcdPx = 28.0f;
const char *kFontDir = "/usr/share/fonts/truetype/dejavu/";

ImU32 grey(int v, int a = 255) { return IM_COL32(v, v, v, a); }

// Palette entries are BGR; dim < 1 darkens, for a lamp that is off.
ImU32 color(const cv::Scalar &bgr, float dim = 1.0f, int a = 255)
{
    return IM_COL32(int(bgr[2] * dim), int(bgr[1] * dim), int(bgr[0] * dim), a);
}

// A plate with a vertical gradient, a lit top-left edge, a shadowed bottom-right edge and a screw in each corner.
void bevel_panel(ImDrawList *dl, ImVec2 a, ImVec2 b)
{
    dl->AddRectFilledMultiColor(a, b, grey(78), grey(78), grey(46), grey(46));
    dl->AddLine({a.x, a.y + 0.5f}, {b.x, a.y + 0.5f}, grey(135));
    dl->AddLine({a.x + 0.5f, a.y}, {a.x + 0.5f, b.y}, grey(120));
    dl->AddLine({a.x, b.y - 0.5f}, {b.x, b.y - 0.5f}, grey(14));
    dl->AddLine({b.x - 0.5f, a.y}, {b.x - 0.5f, b.y}, grey(14));
    for (const ImVec2 c : {ImVec2{a.x + 9, a.y + 9}, ImVec2{b.x - 9, a.y + 9}, ImVec2{a.x + 9, b.y - 9}, ImVec2{b.x - 9, b.y - 9}})
    {
        dl->AddCircleFilled(c, 4.0f, grey(95));
        dl->AddCircle(c, 4.0f, grey(25));
        dl->AddLine({c.x - 2.5f, c.y + 1.5f}, {c.x + 2.5f, c.y - 1.5f}, grey(30), 1.5f);
    }
}

// A recessed well: the top and left edges are in shadow, the lower lip catches the light.
void inset(ImDrawList *dl, ImVec2 a, ImVec2 b, ImU32 fill)
{
    dl->AddRectFilled({a.x - 1, a.y - 1}, {b.x + 1, b.y + 1}, grey(125), 5.0f);
    dl->AddRectFilled(a, b, fill, 4.0f);
    dl->AddRectFilledMultiColor(a, {b.x, a.y + 8}, IM_COL32(0, 0, 0, 140), IM_COL32(0, 0, 0, 140), 0, 0);
    dl->AddRect(a, b, IM_COL32(0, 0, 0, 200), 4.0f);
}

// Lettering pressed into the plate: a dark edge above and a light one below.
void engraved(ImDrawList *dl, ImFont *font, float px, ImVec2 p, const char *text)
{
    dl->AddText(font, px, {p.x, p.y - 1}, grey(10, 200), text);
    dl->AddText(font, px, {p.x, p.y + 1}, grey(150, 160), text);
    dl->AddText(font, px, p, grey(60), text);
}

// A lamp that glows in the class colour when the class is detected and sits dark when it is not.
void led(ImDrawList *dl, ImVec2 c, const cv::Scalar &bgr, bool on)
{
    dl->AddCircleFilled(c, 8.0f, grey(20));
    dl->AddCircle(c, 8.0f, grey(120));
    if (on)
        for (int i = 3; i >= 1; --i)
            dl->AddCircleFilled(c, 6.0f + 2.0f * i, color(bgr, 1.0f, 22));
    dl->AddCircleFilled(c, 5.5f, color(bgr, on ? 1.0f : 0.22f));
    if (on)
        dl->AddCircleFilled({c.x - 1.8f, c.y - 2.0f}, 1.8f, IM_COL32(255, 255, 255, 170));
}

// A lever switch: the knob sits to the right when on. Returns true on the click that flips it.
bool toggle_switch(const char *id, ImVec2 p, bool on)
{
    constexpr ImVec2 size{44, 22};
    ImGui::SetCursorScreenPos(p);
    const bool clicked = ImGui::InvisibleButton(id, size);
    ImDrawList *dl = ImGui::GetWindowDrawList();
    inset(dl, p, {p.x + size.x, p.y + size.y}, on ? IM_COL32(28, 70, 36, 255) : IM_COL32(30, 22, 22, 255));
    const float kx = on ? p.x + size.x - 21 : p.x + 2;
    const ImVec2 k0{kx, p.y + 2}, k1{kx + 19, p.y + size.y - 2};
    dl->AddRectFilledMultiColor(k0, k1, grey(205), grey(205), grey(120), grey(120));
    dl->AddRect(k0, k1, grey(25), 3.0f);
    for (int i = -1; i <= 1; ++i) // grip ridges
        dl->AddLine({(k0.x + k1.x) / 2 + 3.0f * i, k0.y + 4}, {(k0.x + k1.x) / 2 + 3.0f * i, k1.y - 4}, grey(95), 1.5f);
    return clicked;
}

// A roll-up shutter over [a, b]: corrugated slats hang from the top and stack upward from a hazard-striped bottom
// bar, so they travel with the bar as it rises. travel 0 covers the whole screen; 1 leaves just the bar under the drum.
void roller_shutter(ImDrawList *dl, ImVec2 a, ImVec2 b, float travel)
{
    constexpr float rail = 4.0f, bar = 6.0f, slat = 7.0f;
    const float yb = a.y + bar + (1.0f - travel) * (b.y - a.y - bar); // bottom of the bar
    const float x0 = a.x + rail, x1 = b.x - rail;
    dl->PushClipRect({x0, a.y}, {x1, b.y}, true);
    for (float y = yb - bar - slat; y + slat > a.y; y -= slat)
    {
        dl->AddRectFilledMultiColor({x0, y}, {x1, y + slat}, grey(150), grey(150), grey(95), grey(95));
        dl->AddLine({x0, y + 0.5f}, {x1, y + 0.5f}, grey(40));
        dl->AddLine({x0, y + slat - 0.5f}, {x1, y + slat - 0.5f}, grey(185));
    }
    dl->AddRectFilled({x0, yb - bar}, {x1, yb}, IM_COL32(240, 200, 30, 255));
    for (float x = x0 - bar; x < x1; x += 12.0f)
        dl->AddQuadFilled({x, yb}, {x + 6, yb}, {x + 6 + bar, yb - bar}, {x + bar, yb - bar}, grey(20));
    dl->AddRect({x0, yb - bar}, {x1, yb}, grey(20));
    const float cx = (x0 + x1) / 2;
    dl->AddRectFilled({cx - 12, yb - 5}, {cx + 12, yb - 1}, grey(200), 1.5f); // pull handle
    dl->AddRect({cx - 12, yb - 5}, {cx + 12, yb - 1}, grey(30), 1.5f);
    dl->PopClipRect();
    for (const float rx : {a.x, b.x - rail})
    {
        dl->AddRectFilledMultiColor({rx, a.y}, {rx + rail, b.y}, grey(135), grey(100), grey(100), grey(135));
        dl->AddRect({rx, a.y}, {rx + rail, b.y}, grey(25));
    }
}

ImFont *load_font(ImGuiIO &io, const char *file, float px)
{
    const std::string path = std::string(kFontDir) + file;
    if (std::filesystem::exists(path))
        if (ImFont *f = io.Fonts->AddFontFromFileTTF(path.c_str(), px))
            return f;
    return io.Fonts->AddFontDefault();
}

ImVec2 text_size(ImFont *font, float px, const char *text) { return font->CalcTextSizeA(px, FLT_MAX, 0.0f, text); }

// Distance from the top of a line to its baseline at the given size.
float ascent(ImFont *font, float px) { return font->Ascent * px / font->FontSize; }
} // namespace

Gui::Gui(std::vector<std::string> names, double stream_fps) : names_(std::move(names)), stream_fps_(stream_fps)
{
    glfwSetErrorCallback([](int, const char *msg) { std::cerr << "glfw: " << msg << "\n"; });
    if (!glfwInit())
        throw std::runtime_error("Cannot open a window: is DISPLAY set and X access granted (xhost +local:)?");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE); // the size below is what un-maximizing returns to
    window_ = glfwCreateWindow(1280 + int(kSidebarW), 720, "Playback", nullptr, nullptr);
    if (!window_)
    {
        glfwTerminate();
        throw std::runtime_error("Cannot create an OpenGL 3.0 window");
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0); // vsync would hold the sink to the monitor rate
    std::cerr << "GL_RENDERER: " << glGetString(GL_RENDERER) << "\n"; // llvmpipe here means no GPU GL in the container

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    sans_ = load_font(io, "DejaVuSans.ttf", kSansPx);
    mono_ = load_font(io, "DejaVuSansMono.ttf", kMonoPx);
    lcd_ = load_font(io, "DejaVuSansMono-Bold.ttf", kLcdPx);
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
}

Gui::~Gui()
{
    glDeleteTextures(1, &tex_);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window_);
    glfwTerminate();
}

void Gui::upload(const cv::Mat &bgr)
{
    glBindTexture(GL_TEXTURE_2D, tex_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, static_cast<GLint>(bgr.step / 3));
    if (bgr.cols != tex_w_ || bgr.rows != tex_h_)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, bgr.cols, bgr.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, bgr.data);
        tex_w_ = bgr.cols;
        tex_h_ = bgr.rows;
    }
    else
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bgr.cols, bgr.rows, GL_BGR, GL_UNSIGNED_BYTE, bgr.data);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
}

bool Gui::show(Frame &f)
{
    f.stage_ms.render = timed(render_tm_, f.id, "render", [&] {
        glfwPollEvents();
        legend_.update(f.dets, Legend::Clock::now());
        draw_boxes(f.image, f.dets, [&](int class_id) -> std::optional<cv::Scalar> {
            if (const auto slot = legend_.box_slot(class_id))
                return palette()[*slot];
            return std::nullopt;
        });
        upload(f.image);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        const ImVec2 view = ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(view);
        ImGui::Begin("root", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                         ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus);

        // The video keeps its aspect ratio inside the area left of the sidebar.
        const float vw = std::max(view.x - kSidebarW, 1.0f);
        const float scale = std::min(vw / f.image.cols, view.y / f.image.rows);
        const ImVec2 sz{f.image.cols * scale, f.image.rows * scale};
        const ImVec2 p0{(vw - sz.x) / 2, (view.y - sz.y) / 2};
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)tex_, p0, {p0.x + sz.x, p0.y + sz.y});

        draw_sidebar(f, vw, kSidebarW, view.y);
        ImGui::End();

        ImGui::Render();
        int fw, fh;
        glfwGetFramebufferSize(window_, &fw, &fh);
        glViewport(0, 0, fw, fh);
        glClearColor(0.07f, 0.07f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window_);
    });
    return !glfwWindowShouldClose(window_) && glfwGetKey(window_, GLFW_KEY_Q) != GLFW_PRESS &&
           glfwGetKey(window_, GLFW_KEY_ESCAPE) != GLFW_PRESS;
}

void Gui::draw_sidebar(const Frame &f, float x0, float w, float h)
{
    ImDrawList *dl = ImGui::GetWindowDrawList();
    bevel_panel(dl, {x0, 0}, {x0 + w, h});
    const float pad = 18.0f, inner = w - 2 * pad;
    float y = 18.0f;

    // The debug unit is pinned to the bottom, so the legend knows how far down it may reach.
    constexpr float drum_h = 22.0f, screen_h = 42.0f, gap = 18.0f;
    const ImVec2 top{x0 + pad, h - pad - drum_h - screen_h};

    // Readout: green LCD with unlit segments ghosted behind the digits.
    const float lcd_h = 140;
    inset(dl, {x0 + pad, y}, {x0 + pad + inner, y + lcd_h}, IM_COL32(26, 44, 26, 255));
    const auto lcd_row = [&](float ry, const char *label, const std::string &value) {
        // Text is placed by its top edge, so offset the label to put both baselines at ry + the value's ascent.
        const float label_y = ry + ascent(lcd_, kLcdPx) - ascent(sans_, kSansPx);
        dl->AddText(sans_, kSansPx, {x0 + pad + 12, label_y}, IM_COL32(110, 190, 110, 255), label);
        // The unlit segments are the value's own shape with every digit an 8, so they sit exactly under it.
        std::string ghost = value;
        std::replace_if(ghost.begin(), ghost.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }, '8');
        const float tx = x0 + pad + inner - 12 - text_size(lcd_, kLcdPx, value.c_str()).x;
        dl->AddText(lcd_, kLcdPx, {tx, ry}, IM_COL32(120, 255, 120, 28), ghost.c_str());
        dl->AddText(lcd_, kLcdPx, {tx, ry}, IM_COL32(130, 255, 130, 255), value.c_str());
    };
    lcd_row(y + 8, "STREAM FPS", cv::format("%.1f", stream_fps_));
    lcd_row(y + 52, "INFER FPS", cv::format("%.1f", f.fps));
    lcd_row(y + 96, "DET", std::to_string(legend_.shown_count()));
    y += lcd_h + 22;

    // Legend: one row per class the sidebar still lists. The plate stops `gap` above the debug unit and the rows
    // scroll inside it.
    const float row_h = 32;
    const auto &entries = legend_.entries();
    const float content_h = std::max<size_t>(entries.size(), 1) * row_h + 12;
    const float plate_h = std::max(std::min(content_h, top.y - gap - y), row_h + 12);
    inset(dl, {x0 + pad, y}, {x0 + pad + inner, y + plate_h}, grey(30));
    ImGui::SetCursorScreenPos({x0 + pad, y});
    ImGui::PushStyleColor(ImGuiCol_ChildBg, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 6.0f);
    if (ImGui::BeginChild("legend", {inner, plate_h}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground))
    {
        ImDrawList *cdl = ImGui::GetWindowDrawList();
        const ImVec2 origin = ImGui::GetCursorScreenPos(); // moves up as the rows scroll
        if (entries.empty())
            cdl->AddText(sans_, kSansPx, {origin.x + 12, origin.y + 14}, grey(110), "no detections");
        for (size_t i = 0; i < entries.size(); ++i)
        {
            const LegendEntry &e = entries[i];
            const float ry = origin.y + 6 + i * row_h;
            const bool on = e.count > 0;
            const cv::Scalar &c = palette()[e.slot];
            led(cdl, {origin.x + 20, ry + row_h / 2}, c, on);
            const std::string name = e.class_id < static_cast<int>(names_.size()) ? names_[e.class_id] : std::to_string(e.class_id);
            cdl->AddText(sans_, kSansPx, {origin.x + 38, ry + 7}, on ? grey(225) : grey(105), name.c_str());
            const std::string count = on ? "x" + std::to_string(e.count) : "";
            cdl->AddText(sans_, kSansPx, {origin.x + inner - 112, ry + 7}, grey(160), count.c_str());
            const bool shown = legend_.shown(e.class_id);
            if (toggle_switch(("sw" + std::to_string(e.class_id)).c_str(), {origin.x + inner - 56, ry + 5}, shown))
                legend_.set_shown(e.class_id, !shown); // does not touch entries_, so the reference stays valid
        }
        ImGui::SetCursorScreenPos(origin);
        ImGui::Dummy({inner, content_h}); // the content height that drives scrolling
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();

    // Debug: a fixed screen under a roll-up shutter, the drum box above it, pinned to the sidebar's bottom so a growing
    // legend never moves it.
    const ImVec2 screen0{top.x, top.y + drum_h}, screen1{top.x + inner, top.y + drum_h + screen_h};
    ImGui::SetCursorScreenPos(top);
    if (ImGui::InvisibleButton("shutter", {inner, drum_h + screen_h}))
        shutter_open_ = !shutter_open_;
    const float target = shutter_open_ ? 1.0f : 0.0f;
    const float step = ImGui::GetIO().DeltaTime / 0.4f; // 0.4 s to travel
    shutter_ = shutter_ < target ? std::min(shutter_ + step, target) : std::max(shutter_ - step, target);

    inset(dl, screen0, screen1, IM_COL32(14, 16, 14, 255));
    for (size_t i = 0; i < stage_rows_.size(); ++i)
    {
        const bool blank = stage_rows_[i].empty(); // before the first report
        dl->AddText(mono_, kMonoPx, {screen0.x + 12, screen0.y + 8 + i * 13.0f}, blank ? grey(70) : IM_COL32(140, 230, 140, 255),
                    blank ? "--" : stage_rows_[i].c_str());
    }
    roller_shutter(dl, screen0, screen1, shutter_ * shutter_ * (3.0f - 2.0f * shutter_)); // smoothstep

    dl->AddRectFilledMultiColor(top, {top.x + inner, top.y + drum_h}, grey(130), grey(130), grey(72), grey(72));
    dl->AddRect(top, {top.x + inner, top.y + drum_h}, grey(20), 3.0f);
    dl->AddLine({top.x + 1, top.y + 1.5f}, {top.x + inner - 1, top.y + 1.5f}, grey(185));
    for (const float bx : {top.x + 9, top.x + inner - 9})
        dl->AddCircleFilled({bx, top.y + drum_h / 2}, 3.0f, grey(55));
    engraved(dl, sans_, kSansPx, {top.x + 24, top.y + 3}, "DEBUG");
    led(dl, {top.x + inner - 28, top.y + drum_h / 2}, cv::Scalar(0, 170, 255), shutter_ != target); // amber while it moves
}
