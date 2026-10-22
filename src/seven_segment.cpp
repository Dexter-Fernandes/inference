#include "seven_segment.hpp"

float digit_width(float height) { return 0.55f * height; }

float digit_thickness(float height) { return 0.13f * height; }

unsigned segment_mask(char c)
{
    static constexpr unsigned kDigits[10] = {63, 6, 91, 79, 102, 109, 125, 7, 127, 111}; // 0-9
    if (c >= '0' && c <= '9')
        return kDigits[c - '0'];
    return c == '-' ? 1u << 6 : 0u;
}

std::array<Segment, kSegmentCount> segment_shapes(float height)
{
    const float w = digit_width(height), ht = digit_thickness(height) / 2, mid = height / 2;
    const float gap = ht / 2; // keeps neighbouring segments apart
    const auto horizontal = [&](float yc) -> Segment {
        const float xl = ht + gap, xr = w - ht - gap;
        return {{{xl, yc}, {xl + ht, yc - ht}, {xr - ht, yc - ht}, {xr, yc}, {xr - ht, yc + ht}, {xl + ht, yc + ht}}};
    };
    const auto vertical = [&](float xc, float yt, float yb) -> Segment {
        return {{{xc, yt}, {xc + ht, yt + ht}, {xc + ht, yb - ht}, {xc, yb}, {xc - ht, yb - ht}, {xc - ht, yt + ht}}};
    };
    const float top0 = ht + gap, top1 = mid - gap;                 // the upper verticals
    const float bot0 = mid + gap, bot1 = height - ht - gap;        // the lower verticals
    return {horizontal(ht),                  // a
            vertical(w - ht, top0, top1),    // b
            vertical(w - ht, bot0, bot1),    // c
            horizontal(height - ht),         // d
            vertical(ht, bot0, bot1),        // e
            vertical(ht, top0, top1),        // f
            horizontal(mid)};                // g
}
