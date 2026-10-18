#include "pool.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace
{
constexpr auto kBlocked = 50ms;
constexpr auto kWoken = 2s;

TEST(Pool, ReleasedObjectIsLentOutAgainWithItsState)
{
    Pool<int> pool(std::vector<int>{0, 0});
    auto a = pool.try_acquire();
    auto b = pool.try_acquire();
    ASSERT_TRUE(a && b);
    EXPECT_NE(a.get(), b.get());
    *a = 42;
    int *const a_addr = a.get();

    auto a_copy = a;
    a.reset();
    EXPECT_EQ(pool.try_acquire(), nullptr) << "a copy still holds it";

    a_copy.reset();
    auto again = pool.try_acquire();
    ASSERT_NE(again, nullptr);
    EXPECT_EQ(again.get(), a_addr);
    EXPECT_EQ(*again, 42);
}

TEST(Pool, TryAcquireIsEmptyWhenExhausted)
{
    Pool<int> pool(std::vector<int>(3));
    std::vector<std::shared_ptr<int>> held;
    for (int i = 0; i < 3; ++i)
        held.push_back(pool.try_acquire());
    for (const auto &h : held)
        EXPECT_NE(h, nullptr);
    EXPECT_EQ(pool.try_acquire(), nullptr);
}

TEST(Pool, AcquireBlocksUntilAHandleIsReleased)
{
    Pool<int> pool(std::vector<int>(1));
    std::stop_source ss;
    auto held = pool.acquire(ss.get_token());
    ASSERT_NE(held, nullptr);
    int *const addr = held.get();

    auto waiter = std::async(std::launch::async, [&] { return pool.acquire(ss.get_token()); });
    ASSERT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
    held.reset();
    ASSERT_EQ(waiter.wait_for(kWoken), std::future_status::ready);
    EXPECT_EQ(waiter.get().get(), addr);
}

TEST(Pool, StopWakesBlockedAcquireEmptyHanded)
{
    Pool<int> pool(std::vector<int>(1));
    std::stop_source ss;
    auto held = pool.try_acquire();
    auto waiter = std::async(std::launch::async, [&] { return pool.acquire(ss.get_token()); });
    ASSERT_EQ(waiter.wait_for(kBlocked), std::future_status::timeout);
    ss.request_stop();
    ASSERT_EQ(waiter.wait_for(kWoken), std::future_status::ready);
    EXPECT_EQ(waiter.get(), nullptr);
}

TEST(Pool, ThreadsSharingThePoolNeverHoldTheSameObject)
{
    constexpr int kObjects = 3, kThreads = 6, kRounds = 5000;
    Pool<int> pool(std::vector<int>(kObjects, 0)); // each object counts its current holders
    std::stop_source ss;
    {
        std::vector<std::jthread> threads;
        for (int t = 0; t < kThreads; ++t)
            threads.emplace_back([&] {
                for (int i = 0; i < kRounds; ++i)
                {
                    auto h = pool.acquire(ss.get_token());
                    ASSERT_EQ(++*h, 1);
                    --*h;
                }
            });
    }
    std::vector<std::shared_ptr<int>> all;
    for (int i = 0; i < kObjects; ++i)
        all.push_back(pool.try_acquire());
    for (const auto &h : all)
        ASSERT_NE(h, nullptr) << "every object came back";
    EXPECT_EQ(pool.try_acquire(), nullptr);
}
} // namespace
