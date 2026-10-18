#include "bounded_queue.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace
{
constexpr auto kBlocked = 50ms; // long enough to see a call is still waiting
constexpr auto kWoken = 2s;     // generous bound for a woken call to return

TEST(BoundedQueue, PopsInPushOrder)
{
    BoundedQueue<int> q(3);
    std::stop_source ss;
    for (int i : {1, 2, 3})
        ASSERT_TRUE(q.push(i, ss.get_token()));
    for (int i : {1, 2, 3})
        EXPECT_EQ(q.pop(ss.get_token()), i);
}

TEST(BoundedQueue, TryPushFailsWhenFull)
{
    BoundedQueue<int> q(2);
    std::stop_source ss;
    EXPECT_TRUE(q.try_push(1));
    EXPECT_TRUE(q.try_push(2));
    EXPECT_FALSE(q.try_push(3));
    EXPECT_EQ(q.pop(ss.get_token()), 1);
    EXPECT_TRUE(q.try_push(4));
    EXPECT_EQ(q.pop(ss.get_token()), 2);
    EXPECT_EQ(q.pop(ss.get_token()), 4);
}

TEST(BoundedQueue, CloseDrainsThenPopsEmpty)
{
    BoundedQueue<int> q(4);
    std::stop_source ss;
    q.push(1, ss.get_token());
    q.push(2, ss.get_token());
    q.close();
    EXPECT_FALSE(q.push(3, ss.get_token()));
    EXPECT_FALSE(q.try_push(3));
    EXPECT_EQ(q.pop(ss.get_token()), 1);
    EXPECT_EQ(q.pop(ss.get_token()), 2);
    EXPECT_EQ(q.pop(ss.get_token()), std::nullopt);
}

TEST(BoundedQueue, CloseWakesBlockedPop)
{
    BoundedQueue<int> q(1);
    std::stop_source ss;
    auto popped = std::async(std::launch::async, [&] { return q.pop(ss.get_token()); });
    ASSERT_EQ(popped.wait_for(kBlocked), std::future_status::timeout);
    q.close();
    ASSERT_EQ(popped.wait_for(kWoken), std::future_status::ready);
    EXPECT_EQ(popped.get(), std::nullopt);
}

TEST(BoundedQueue, StopWakesBlockedPop)
{
    BoundedQueue<int> q(1);
    std::stop_source ss;
    auto popped = std::async(std::launch::async, [&] { return q.pop(ss.get_token()); });
    ASSERT_EQ(popped.wait_for(kBlocked), std::future_status::timeout);
    ss.request_stop();
    ASSERT_EQ(popped.wait_for(kWoken), std::future_status::ready);
    EXPECT_EQ(popped.get(), std::nullopt);
}

TEST(BoundedQueue, StopWakesBlockedPushAndDropsItsItem)
{
    BoundedQueue<int> q(1);
    std::stop_source ss, other;
    ASSERT_TRUE(q.push(1, ss.get_token()));
    auto pushed = std::async(std::launch::async, [&] { return q.push(2, ss.get_token()); });
    ASSERT_EQ(pushed.wait_for(kBlocked), std::future_status::timeout);
    ss.request_stop();
    ASSERT_EQ(pushed.wait_for(kWoken), std::future_status::ready);
    EXPECT_FALSE(pushed.get());
    q.close();
    EXPECT_EQ(q.pop(other.get_token()), 1);
    EXPECT_EQ(q.pop(other.get_token()), std::nullopt);
}

TEST(BoundedQueue, ManyProducersAndConsumersDeliverEachItemOnce)
{
    constexpr int kProducers = 4, kConsumers = 3, kPerProducer = 20000;
    BoundedQueue<int> q(4); // small, so both sides block often
    std::stop_source ss;

    std::vector<std::vector<int>> got(kConsumers);
    {
        std::vector<std::jthread> consumers;
        for (int c = 0; c < kConsumers; ++c)
            consumers.emplace_back([&, c] {
                while (auto v = q.pop(ss.get_token()))
                    got[c].push_back(*v);
            });
        {
            std::vector<std::jthread> producers;
            for (int p = 0; p < kProducers; ++p)
                producers.emplace_back([&, p] {
                    for (int i = 0; i < kPerProducer; ++i)
                        ASSERT_TRUE(q.push(p * kPerProducer + i, ss.get_token()));
                });
        }
        q.close();
    }

    std::vector<int> all;
    for (const auto &g : got)
    {
        // FIFO: one consumer sees each producer's items in the order they were pushed
        for (int p = 0; p < kProducers; ++p)
        {
            std::vector<int> mine;
            std::copy_if(g.begin(), g.end(), std::back_inserter(mine),
                         [&](int v) { return v / kPerProducer == p; });
            EXPECT_TRUE(std::is_sorted(mine.begin(), mine.end())) << "producer " << p;
        }
        all.insert(all.end(), g.begin(), g.end());
    }
    std::sort(all.begin(), all.end());
    std::vector<int> want(kProducers * kPerProducer);
    std::iota(want.begin(), want.end(), 0);
    EXPECT_EQ(all, want);
}
} // namespace
