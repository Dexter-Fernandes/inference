#pragma once

#include <array>

struct Pt
{
    float x, y;
};

// One segment is a hexagon with pointed ends, listed clockwise on a y-down screen.
using Segment = std::array<Pt, 6>;

// Segment i of a digit cell: a top, b top right, c bottom right, d bottom, e bottom left, f top left, g middle.
constexpr int kSegmentCount = 7;

float digit_width(float height);
float digit_thickness(float height);

// Bit i is set when segment i is lit for c. Digits and '-' have a form; anything else, a space included, lights nothing.
unsigned segment_mask(char c);

// The segments of a cell `height` tall and digit_width(height) wide, with its origin at the top left and no slant.
std::array<Segment, kSegmentCount> segment_shapes(float height);
