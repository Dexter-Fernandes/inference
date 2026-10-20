#include "det_ring.hpp"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace
{
std::string unique_name()
{
    static int n = 0;
    // Containers have their own pid namespace but share /dev/shm under ipc: host, so the pid alone can collide.
    const auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    return "/det_ring_test_" + std::to_string(getpid()) + "_" + std::to_string(t) + "_" + std::to_string(n++);
}

// Every word depends on n, so a frame mixing two publications cannot match either.
DetFrame make(uint64_t n)
{
    DetFrame f{};
    f.frame_id = n;
    f.captured_ns = static_cast<int64_t>(n * 3);
    f.published_ns = static_cast<int64_t>(n * 5);
    f.count = static_cast<uint32_t>(n % (kMaxDets + 1));
    for (uint32_t i = 0; i < kMaxDets; ++i)
        f.dets[i] = {static_cast<int32_t>(n + i), static_cast<float>(n % 1000) + i, static_cast<int32_t>(n * 7 + i),
                     static_cast<int32_t>(n * 11 + i), static_cast<int32_t>(n * 13 + i), static_cast<int32_t>(n * 17 + i)};
    return f;
}

bool same(const DetFrame &a, const DetFrame &b) { return std::memcmp(&a, &b, sizeof a) == 0; }

TEST(DetRing, ReadsBackWhatWasPublished)
{
    const std::string name = unique_name();
    DetWriter w(name);
    DetReader r(name);
    EXPECT_EQ(r.published(), 0u);
    DetFrame out;
    EXPECT_EQ(r.read(0, out), ReadResult::Pending);

    for (uint64_t n = 0; n < 3; ++n)
        w.publish(make(n));
    EXPECT_EQ(r.published(), 3u);
    for (uint64_t n = 0; n < 3; ++n)
    {
        ASSERT_EQ(r.read(n, out), ReadResult::Ok) << n;
        EXPECT_TRUE(same(out, make(n))) << n;
    }
    EXPECT_EQ(r.read(3, out), ReadResult::Pending);
}

TEST(DetRing, LapsAReaderAfterTheRingWraps)
{
    const std::string name = unique_name();
    DetWriter w(name);
    DetReader r(name);
    for (uint64_t n = 0; n < kRingSlots + 1; ++n)
        w.publish(make(n));
    DetFrame out;
    EXPECT_EQ(r.read(0, out), ReadResult::Lapped);
    for (uint64_t n = 1; n <= kRingSlots; ++n)
    {
        ASSERT_EQ(r.read(n, out), ReadResult::Ok) << n;
        EXPECT_TRUE(same(out, make(n))) << n;
    }
}

TEST(DetRing, ReaderKeepsTheLastFramesAfterTheWriterIsGone)
{
    const std::string name = unique_name();
    std::optional<DetWriter> w(std::in_place, name);
    DetReader r(name);
    w->publish(make(0));
    w.reset();
    DetFrame out;
    ASSERT_EQ(r.read(0, out), ReadResult::Ok);
    EXPECT_TRUE(same(out, make(0)));
    EXPECT_THROW(DetReader{name}, std::runtime_error) << "the name is unlinked";
}

TEST(DetRing, WriterReplacesAStaleFile)
{
    const std::string name = unique_name();
    const int stale = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
    ASSERT_GE(stale, 0);
    close(stale);
    EXPECT_NO_THROW(DetWriter{name});
}

TEST(DetRing, ReaderRejectsAFileThatIsNotARing)
{
    const std::string name = unique_name();
    EXPECT_THROW(DetReader{name}, std::runtime_error) << "missing";

    const int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(ftruncate(fd, 100), 0);
    EXPECT_THROW(DetReader{name}, std::runtime_error) << "wrong size";
    close(fd);
    shm_unlink(name.c_str());

    DetWriter w(name);
    const int rw = shm_open(name.c_str(), O_RDWR, 0);
    ASSERT_GE(rw, 0);
    const uint64_t zero = 0; // the magic is the first word of the file
    ASSERT_EQ(pwrite(rw, &zero, sizeof zero, 0), static_cast<ssize_t>(sizeof zero));
    close(rw);
    EXPECT_THROW(DetReader{name}, std::runtime_error) << "bad magic";
}

TEST(DetRing, ReaderKeepsACorruptCountInsideTheFrame)
{
    const std::string name = unique_name();
    DetWriter w(name);
    DetReader r(name);
    DetFrame bad = make(0);
    bad.count = 1000000;
    w.publish(bad);
    DetFrame out;
    ASSERT_EQ(r.read(0, out), ReadResult::Ok);
    EXPECT_EQ(out.count, kMaxDets);
}

// The torn-frame count is the ordering evidence. ThreadSanitizer does not model fences and sees the writer and
// each reader at different addresses, so it only checks the rest of the test. On x86 only the seq recheck
// is proven here: deleting a fence still passes, so the fences rest on the seqlock argument.
TEST(DetRing, ConcurrentReadersNeverSeeATornFrame)
{
    constexpr uint64_t kPublications = 200000;
    constexpr int kReaders = 2;
    const std::string name = unique_name();
    DetWriter w(name);
    DetReader r(name);

    std::atomic<uint64_t> torn{0}, ok{0};
    {
        std::vector<std::jthread> readers;
        for (int i = 0; i < kReaders; ++i)
            readers.emplace_back([&] {
                DetFrame out;
                uint64_t next = 0, mine = 0;
                while (next < kPublications)
                    switch (r.read(next, out))
                    {
                    case ReadResult::Ok:
                        if (!same(out, make(next)))
                            torn.fetch_add(1, std::memory_order_relaxed);
                        ++mine;
                        ++next;
                        break;
                    case ReadResult::Pending:
                        std::this_thread::yield();
                        break;
                    case ReadResult::Lapped:
                        next = std::max(next + 1, r.published() - 1); // published() is at least 8 once a slot is lapped
                        break;
                    }
                ok.fetch_add(mine, std::memory_order_relaxed);
            });
        for (uint64_t n = 0; n < kPublications; ++n)
        {
            w.publish(make(n));
            if (n % 16 == 0)
                std::this_thread::yield(); // lets readers land a stable slot between overwrites
        }
    }
    EXPECT_EQ(torn.load(), 0u);
    EXPECT_GT(ok.load(), 1000u) << "a run of lapped reads proves nothing";
}
} // namespace
