#pragma once

#include <atomic>
#include <csignal>

inline std::atomic<bool> g_stop{false};
static_assert(std::atomic<bool>::is_always_lock_free);

// SIGINT and SIGTERM set g_stop.
inline void install_signal_handlers()
{
    struct sigaction sa{};
    sa.sa_handler = [](int) { g_stop.store(true, std::memory_order_relaxed); };
    sigemptyset(&sa.sa_mask);
    for (const int sig : {SIGINT, SIGTERM})
        sigaction(sig, &sa, nullptr);
}
