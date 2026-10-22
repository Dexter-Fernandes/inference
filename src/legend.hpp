#pragma once

#include "common.hpp"

#include <chrono>
#include <optional>
#include <set>
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
// kLinger has passed; its colour slot is free for another class only once it has left.
class Legend
{
public:
    using Clock = std::chrono::steady_clock;
    static constexpr std::chrono::seconds kLinger{15};

    void update(const std::vector<Detection> &dets, Clock::time_point now);

    const std::vector<LegendEntry> &entries() const { return entries_; } // by class id

    // Palette slot to draw class_id's boxes with; nullopt when the class is not listed or is hidden.
    std::optional<int> box_slot(int class_id) const;

    bool hidden(int class_id) const { return hidden_.count(class_id) > 0; }
    void set_hidden(int class_id, bool hide); // outlives the entry, so a returning class stays hidden

private:
    std::vector<LegendEntry> entries_;
    std::set<int> hidden_;
};
