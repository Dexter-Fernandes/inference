#include "det_ring.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace
{
constexpr uint64_t kMagic = 0x44455452494e4731; // "DETRING1"
constexpr uint32_t kVersion = 1;
constexpr size_t kWords = sizeof(DetFrame) / 8;

static_assert(std::atomic<uint64_t>::is_always_lock_free); // a load must not write: the reader's mapping is PROT_READ

[[noreturn]] void fail(const char *call, const std::string &name)
{
    throw std::runtime_error(std::string(call) + " " + name + ": " + std::strerror(errno));
}

struct Fd
{
    int fd;
    ~Fd() { close(fd); }
};

struct Unlinker
{
    const std::string &name;
    bool armed = true;
    ~Unlinker()
    {
        if (armed)
            shm_unlink(name.c_str());
    }
};
} // namespace

// Publication n lives in slot n % kRingSlots. Its seq is 2n+1 while the writer fills it and 2n+2 once
// stable, so a reader tells "not yet", "stable" and "overwritten" apart from seq alone.
struct DetRing
{
    struct Slot
    {
        std::atomic<uint64_t> seq;
        std::atomic<uint64_t> words[kWords]; // a DetFrame, one relaxed word at a time; the seq fences order them
    };

    std::atomic<uint64_t> magic; // stored last, so a reader never sees a half-initialised header
    uint32_t version, slots;
    std::atomic<uint64_t> write_seq; // publications completed
    Slot slot[kRingSlots];
};

DetWriter::DetWriter(std::string name) : name_(std::move(name))
{
    shm_unlink(name_.c_str());
    Fd fd{shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600)}; // det_sub runs as the same uid
    if (fd.fd < 0)
        fail("shm_open", name_);
    Unlinker created{name_}; // a half-made file would otherwise outlive the failed constructor
    if (ftruncate(fd.fd, sizeof(DetRing)) < 0)
        fail("ftruncate", name_);
    void *p = mmap(nullptr, sizeof(DetRing), PROT_READ | PROT_WRITE, MAP_SHARED, fd.fd, 0);
    if (p == MAP_FAILED)
        fail("mmap", name_);
    ring_ = new (p) DetRing{};
    ring_->version = kVersion;
    ring_->slots = kRingSlots;
    ring_->magic.store(kMagic, std::memory_order_release);
    created.armed = false;
}

DetWriter::~DetWriter()
{
    munmap(ring_, sizeof(DetRing));
    shm_unlink(name_.c_str());
}

void DetWriter::publish(const DetFrame &f)
{
    uint64_t words[kWords];
    std::memcpy(words, &f, sizeof f);

    const uint64_t n = next_++;
    DetRing::Slot &s = ring_->slot[n % kRingSlots];
    s.seq.store(2 * n + 1, std::memory_order_relaxed);
    // Orders the odd seq before the payload stores; the payload stores are relaxed.
    std::atomic_thread_fence(std::memory_order_release);
    for (size_t i = 0; i < kWords; ++i)
        s.words[i].store(words[i], std::memory_order_relaxed);
    s.seq.store(2 * n + 2, std::memory_order_release);
    ring_->write_seq.store(n + 1, std::memory_order_release);
}

DetReader::DetReader(const std::string &name)
{
    Fd fd{shm_open(name.c_str(), O_RDONLY, 0)};
    if (fd.fd < 0)
        fail("shm_open", name);
    struct stat st;
    if (fstat(fd.fd, &st) < 0)
        fail("fstat", name);
    if (static_cast<size_t>(st.st_size) != sizeof(DetRing))
        throw std::runtime_error(name + " is not a detection ring: wrong size");
    void *p = mmap(nullptr, sizeof(DetRing), PROT_READ, MAP_SHARED, fd.fd, 0);
    if (p == MAP_FAILED)
        fail("mmap", name);
    ring_ = static_cast<const DetRing *>(p);
    if (ring_->magic.load(std::memory_order_acquire) != kMagic || ring_->version != kVersion || ring_->slots != kRingSlots)
    {
        munmap(p, sizeof(DetRing));
        throw std::runtime_error(name + " is not a detection ring: wrong magic or version");
    }
}

DetReader::~DetReader() { munmap(const_cast<DetRing *>(ring_), sizeof(DetRing)); }

uint64_t DetReader::published() const { return ring_->write_seq.load(std::memory_order_acquire); }

ReadResult DetReader::read(uint64_t n, DetFrame &out) const
{
    const DetRing::Slot &s = ring_->slot[n % kRingSlots];
    const uint64_t stable = 2 * n + 2;
    const uint64_t before = s.seq.load(std::memory_order_acquire);
    if (before < stable)
        return ReadResult::Pending;
    if (before > stable)
        return ReadResult::Lapped;

    uint64_t words[kWords];
    for (size_t i = 0; i < kWords; ++i)
        words[i] = s.words[i].load(std::memory_order_relaxed);
    // Orders the payload loads before the recheck. Without it they could be reordered after it, past a
    // writer that has already started on this slot.
    std::atomic_thread_fence(std::memory_order_acquire);
    if (s.seq.load(std::memory_order_relaxed) != before)
        return ReadResult::Lapped;

    std::memcpy(&out, words, sizeof out);
    // The file is another process's, and anyone can create the name: keep count inside dets for every caller.
    out.count = std::min(out.count, kMaxDets);
    return ReadResult::Ok;
}
