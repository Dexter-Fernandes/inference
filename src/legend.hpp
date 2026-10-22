#pragma once

#include "common.hpp"

#include <chrono>
#include <optional>
#include <set>
#include <utility>
#include <vector>

constexpr int kPaletteSize = 12;

struct LegendEntry
{
    int class_id;
    int slot;  // palette index, kept until the entry leaves
    int count; // boxes in the latest frame; 0 = not detected now
    std::chrono::steady_clock::time_point last_seen;
};

// The classes the sidebar lists. A class stays after it stops being detected, with count 0, until
// kLinger has passed; its colour slot is free for another class only once it has left. Each class has a
// switch for its boxes; the classes in the constructor's set start on and every other class starts off.
class Legend
{
public:
    using Clock = std::chrono::steady_clock;
    static constexpr std::chrono::seconds kLinger{15};

    explicit Legend(std::set<int> shown = {}) : shown_(std::move(shown)) {}

    void update(const std::vector<Detection> &dets, Clock::time_point now);

    const std::vector<LegendEntry> &entries() const { return entries_; } // by class id

    // Palette slot to draw class_id's boxes with; nullopt when the class is not listed or its switch is off.
    std::optional<int> box_slot(int class_id) const;

    // Boxes of the switched-on classes in the latest frame, the sum of their counts.
    int shown_count() const;

    bool shown(int class_id) const { return shown_.count(class_id) > 0; }
    void set_shown(int class_id, bool on); // outlives the entry, so a returning class keeps its switch

private:
    std::vector<LegendEntry> entries_;
    std::set<int> shown_;
};
