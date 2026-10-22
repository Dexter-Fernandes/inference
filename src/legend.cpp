#include "legend.hpp"

#include <algorithm>
#include <map>

void Legend::update(const std::vector<Detection> &dets, Clock::time_point now)
{
    std::map<int, int> counts;
    for (const auto &d : dets)
        ++counts[d.class_id];

    for (auto &e : entries_)
    {
        const auto it = counts.find(e.class_id);
        e.count = it == counts.end() ? 0 : it->second;
        if (e.count > 0)
            e.last_seen = now;
    }
    std::erase_if(entries_, [&](const LegendEntry &e) { return now - e.last_seen >= kLinger; });

    std::set<int> used;
    for (const auto &e : entries_)
        used.insert(e.slot);
    for (const auto &[class_id, count] : counts)
    {
        if (std::any_of(entries_.begin(), entries_.end(), [&](const LegendEntry &e) { return e.class_id == class_id; }))
            continue;
        int slot = 0;
        while (slot < kPaletteSize && used.count(slot))
            ++slot;
        if (slot == kPaletteSize) // ponytail: more than kPaletteSize listed classes share colours; widen the palette if it matters
            slot = class_id % kPaletteSize;
        used.insert(slot);
        entries_.push_back({class_id, slot, count, now});
    }
    std::sort(entries_.begin(), entries_.end(),
              [](const LegendEntry &a, const LegendEntry &b) { return a.class_id < b.class_id; });
}

std::optional<int> Legend::box_slot(int class_id) const
{
    if (!shown(class_id))
        return std::nullopt;
    for (const auto &e : entries_)
        if (e.class_id == class_id)
            return e.slot;
    return std::nullopt;
}

int Legend::shown_count() const
{
    int n = 0;
    for (const auto &e : entries_)
        if (shown(e.class_id))
            n += e.count;
    return n;
}

void Legend::set_shown(int class_id, bool on)
{
    if (on)
        shown_.insert(class_id);
    else
        shown_.erase(class_id);
}
