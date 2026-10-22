#include "legend.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <set>

namespace
{
using std::chrono::seconds;
const Legend::Clock::time_point t0{};

Detection det(int class_id) { return {class_id, 0.9f, {0, 0, 10, 10}}; }
} // namespace

TEST(Legend, NewClassGetsASlotAndACount)
{
    Legend l;
    l.update({det(2), det(2), det(7)}, t0);
    ASSERT_EQ(l.entries().size(), 2u);
    EXPECT_EQ(l.entries()[0].class_id, 2);
    EXPECT_EQ(l.entries()[0].count, 2);
    EXPECT_EQ(l.entries()[1].count, 1);
    EXPECT_NE(l.entries()[0].slot, l.entries()[1].slot);
}

TEST(Legend, EntriesAreSortedByClassId)
{
    Legend l;
    l.update({det(9), det(1), det(5)}, t0);
    l.update({det(3)}, t0 + seconds(1));
    int prev = -1;
    for (const auto &e : l.entries())
    {
        EXPECT_GT(e.class_id, prev);
        prev = e.class_id;
    }
}

TEST(Legend, AbsentClassStaysGreyedForFifteenSeconds)
{
    Legend l;
    l.update({det(1)}, t0);
    l.update({}, t0 + seconds(14));
    ASSERT_EQ(l.entries().size(), 1u);
    EXPECT_EQ(l.entries()[0].count, 0);
    l.update({}, t0 + seconds(15));
    EXPECT_TRUE(l.entries().empty());
}

TEST(Legend, SeenAgainRestartsTheClock)
{
    Legend l;
    l.update({det(1)}, t0);
    l.update({det(1)}, t0 + seconds(10));
    l.update({}, t0 + seconds(20));
    ASSERT_EQ(l.entries().size(), 1u);
    EXPECT_EQ(l.entries()[0].count, 0);
    EXPECT_EQ(l.entries()[0].last_seen, t0 + seconds(10));
}

TEST(Legend, SlotIsKeptWhileGreyedAndFreedOnLeaving)
{
    Legend l;
    l.update({det(1)}, t0);
    const int slot = l.entries()[0].slot;
    l.update({det(2)}, t0 + seconds(1)); // class 1 greyed, still holds its slot
    ASSERT_EQ(l.entries().size(), 2u);
    EXPECT_NE(l.entries()[1].slot, slot);
    l.update({det(3)}, t0 + seconds(15)); // class 1 has left (class 2 has not), so 3 takes the lowest free slot
    EXPECT_EQ(l.entries()[0].class_id, 2);
    EXPECT_EQ(l.entries()[1].class_id, 3);
    EXPECT_EQ(l.entries()[1].slot, slot);
}

TEST(Legend, ListedClassesNeverShareASlot)
{
    Legend l;
    std::vector<Detection> dets;
    for (int c = 0; c < kPaletteSize; ++c)
        dets.push_back(det(c * 5)); // class % 12 would collide for several of these
    l.update(dets, t0);
    std::set<int> slots;
    for (const auto &e : l.entries())
        slots.insert(e.slot);
    EXPECT_EQ(slots.size(), static_cast<size_t>(kPaletteSize));
}

TEST(Legend, OnlyTheDefaultClassesStartShown)
{
    Legend l({0});
    l.update({det(0), det(4)}, t0);
    EXPECT_TRUE(l.box_slot(0).has_value());
    EXPECT_FALSE(l.box_slot(4).has_value());
}

TEST(Legend, ShownStateOutlivesTheEntry)
{
    Legend l;
    l.update({det(4)}, t0);
    EXPECT_FALSE(l.box_slot(4).has_value());
    l.set_shown(4, true);
    EXPECT_TRUE(l.box_slot(4).has_value());
    l.update({}, t0 + seconds(20));
    EXPECT_TRUE(l.entries().empty());
    l.update({det(4)}, t0 + seconds(21));
    EXPECT_TRUE(l.shown(4));
    EXPECT_TRUE(l.box_slot(4).has_value());
    l.set_shown(4, false);
    EXPECT_FALSE(l.box_slot(4).has_value());
}

TEST(Legend, ShownCountSumsOnlyTheSwitchedOnClasses)
{
    Legend l({0});
    l.update({det(0), det(0), det(0), det(2), det(2)}, t0);
    EXPECT_EQ(l.shown_count(), 3);
    l.set_shown(2, true);
    EXPECT_EQ(l.shown_count(), 5);
    l.update({det(0), det(0), det(0)}, t0 + seconds(1)); // class 2 is greyed, count 0
    EXPECT_EQ(l.shown_count(), 3);
}

TEST(Legend, UnlistedClassHasNoBoxSlot)
{
    Legend l;
    EXPECT_FALSE(l.box_slot(0).has_value());
}
