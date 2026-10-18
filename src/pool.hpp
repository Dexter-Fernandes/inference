#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <stop_token>
#include <vector>

// A fixed set of objects lent out as shared_ptrs; an object returns to the pool when its last
// handle is dropped. The pool must outlive every handle it gave out.
template <class T>
class Pool
{
public:
    explicit Pool(std::vector<T> items) : items_(std::move(items))
    {
        free_.reserve(items_.size()); // so returning a handle never allocates
        for (T &t : items_)
            free_.push_back(&t);
    }
    Pool(const Pool &) = delete;
    Pool &operator=(const Pool &) = delete;

    // Blocks until an object is free. Empty once stop is requested.
    std::shared_ptr<T> acquire(std::stop_token st)
    {
        T *t;
        {
            std::unique_lock lk(m_);
            if (!cv_.wait(lk, st, [&] { return !free_.empty(); }) || st.stop_requested())
                return nullptr;
            t = pop_free();
        }
        return handle(t);
    }

    // Empty if every object is lent out.
    std::shared_ptr<T> try_acquire()
    {
        T *t;
        {
            std::lock_guard lk(m_);
            if (free_.empty())
                return nullptr;
            t = pop_free();
        }
        return handle(t);
    }

private:
    T *pop_free()
    {
        T *t = free_.back();
        free_.pop_back();
        return t;
    }

    // Outside the lock: if allocating the control block throws, the deleter runs and takes it.
    std::shared_ptr<T> handle(T *t)
    {
        return std::shared_ptr<T>(t, [this](T *p) { give_back(p); });
    }

    void give_back(T *p)
    {
        {
            std::lock_guard lk(m_);
            free_.push_back(p);
        }
        cv_.notify_one();
    }

    std::vector<T> items_; // never resized, so the pointers stay valid
    std::vector<T *> free_;
    std::mutex m_;
    std::condition_variable_any cv_;
};
