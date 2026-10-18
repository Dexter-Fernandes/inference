#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stop_token>
#include <utility>

// Fixed-capacity FIFO between threads. A stop request wakes every waiter and ends it empty-handed;
// close() is the graceful end: pushes fail, pops drain what is left and then return empty.
template <class T>
class BoundedQueue
{
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

    // Blocks while full. False if stop was requested or the queue is closed (v is dropped).
    bool push(T v, std::stop_token st)
    {
        std::unique_lock lk(m_);
        if (!cv_.wait(lk, st, [&] { return closed_ || q_.size() < capacity_; }) || st.stop_requested() || closed_)
            return false;
        q_.push_back(std::move(v));
        lk.unlock();
        cv_.notify_all();
        return true;
    }

    // Never blocks. False if full or closed (v is dropped).
    bool try_push(T v)
    {
        {
            std::lock_guard lk(m_);
            if (closed_ || q_.size() >= capacity_)
                return false;
            q_.push_back(std::move(v));
        }
        cv_.notify_all();
        return true;
    }

    // Blocks while empty. Empty once stop is requested, or once the queue is closed and drained.
    std::optional<T> pop(std::stop_token st)
    {
        std::unique_lock lk(m_);
        if (!cv_.wait(lk, st, [&] { return closed_ || !q_.empty(); }) || st.stop_requested() || q_.empty())
            return std::nullopt;
        std::optional<T> v(std::move(q_.front()));
        q_.pop_front();
        lk.unlock();
        cv_.notify_all();
        return v;
    }

    void close()
    {
        {
            std::lock_guard lk(m_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    const size_t capacity_;
    std::deque<T> q_;
    bool closed_ = false;
    std::mutex m_;
    std::condition_variable_any cv_; // one cv for both directions, hence notify_all
};
