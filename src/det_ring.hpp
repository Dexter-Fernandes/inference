#pragma once

#include <cstdint>
#include <string>
#include <type_traits>

// One detection as it sits in shared memory: plain ints and floats, no OpenCV types, so another
// process can read it with this header alone.
struct ShmDetection
{
    int32_t class_id;
    float conf;
    int32_t x, y, w, h;
};

constexpr uint32_t kMaxDets = 128;

// One frame's detections. Fixed size and trivially copyable, so a reader can copy it out of the ring
// word by word. Detections past kMaxDets are the caller's to drop.
struct DetFrame
{
    uint64_t frame_id; // the sink's Frame::id: counts frames the pipeline delivered, not capture drops
    int64_t captured_ns, published_ns; // steady_clock, which containers on one host share
    uint32_t count, pad;               // count <= kMaxDets
    ShmDetection dets[kMaxDets];
};
static_assert(std::is_trivially_copyable_v<DetFrame> && sizeof(DetFrame) % 8 == 0);

constexpr uint32_t kRingSlots = 8;

enum class ReadResult
{
    Ok,
    Pending, // not published yet
    Lapped   // overwritten, or being overwritten, by a later publication
};

struct DetRing;

// Single writer, any number of readers in other processes, none of them blocking the writer. The ring
// keeps the last kRingSlots frames; a reader that falls behind gets Lapped, never stale data.
//
// ponytail: a producer restart needs a det_sub restart. The reader keeps the old, unlinked mapping and
// waits on it forever. A reader that re-opened on a changed inode would fix that.
class DetWriter
{
public:
    // Replaces a stale file a crashed writer left behind; readers of the old one keep a valid mapping.
    // Throws std::runtime_error naming the failed call.
    explicit DetWriter(std::string name);

    // Unmaps and unlinks the name. Readers already mapped keep reading the last frames.
    ~DetWriter();

    DetWriter(const DetWriter &) = delete;
    DetWriter &operator=(const DetWriter &) = delete;

    // One thread only. Never blocks.
    void publish(const DetFrame &f);

private:
    std::string name_;
    DetRing *ring_;
    uint64_t next_ = 0;
};

class DetReader
{
public:
    // Throws std::runtime_error if the name is missing or holds anything but a ring of this layout.
    explicit DetReader(const std::string &name);
    ~DetReader();

    DetReader(const DetReader &) = delete;
    DetReader &operator=(const DetReader &) = delete;

    // Publications so far; publication n is the n-th (0-based) frame the writer published.
    uint64_t published() const;

    // Copies publication n into out. out is unspecified unless Ok.
    ReadResult read(uint64_t n, DetFrame &out) const;

private:
    const DetRing *ring_;
};
