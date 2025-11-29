#pragma once

// ============================================================================
// ArgusTrader Lock-Free Ring Buffer
// ============================================================================
// Single-producer, single-consumer (SPSC) ring buffer for high-performance
// market data handling. Lock-free for minimal latency.
// ============================================================================

#include <atomic>
#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace argus {

// Cache line size for padding to avoid false sharing
constexpr size_t CACHE_LINE_SIZE = 64;

template <typename T, size_t Capacity>
class RingBuffer {
    static_assert(Capacity > 0, "Capacity must be greater than 0");
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    RingBuffer() : head_(0), tail_(0) {
        // Zero-initialize buffer
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i] = T{};
        }
    }

    // Non-copyable, non-movable
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
    RingBuffer(RingBuffer&&) = delete;
    RingBuffer& operator=(RingBuffer&&) = delete;

    // Try to push an item (producer side)
    // Returns true if successful, false if buffer is full
    bool try_push(const T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (current_head + 1) & (Capacity - 1);

        // Check if buffer is full
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }

        buffer_[current_head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    // Push with overwrite (drops oldest if full)
    void push_overwrite(const T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (current_head + 1) & (Capacity - 1);

        // If full, advance tail (drop oldest)
        if (next_head == tail_.load(std::memory_order_acquire)) {
            tail_.store((tail_.load(std::memory_order_relaxed) + 1) & (Capacity - 1),
                       std::memory_order_release);
        }

        buffer_[current_head] = item;
        head_.store(next_head, std::memory_order_release);
    }

    // Try to pop an item (consumer side)
    // Returns std::nullopt if buffer is empty
    std::optional<T> try_pop() {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);

        // Check if buffer is empty
        if (current_tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }

        T item = buffer_[current_tail];
        tail_.store((current_tail + 1) & (Capacity - 1), std::memory_order_release);
        return item;
    }

    // Peek at front item without removing
    std::optional<T> peek() const {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);

        if (current_tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }

        return buffer_[current_tail];
    }

    // Check if buffer is empty
    bool empty() const {
        return tail_.load(std::memory_order_relaxed) ==
               head_.load(std::memory_order_relaxed);
    }

    // Check if buffer is full
    bool full() const {
        const size_t next_head = (head_.load(std::memory_order_relaxed) + 1) & (Capacity - 1);
        return next_head == tail_.load(std::memory_order_relaxed);
    }

    // Get current size
    size_t size() const {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_relaxed);
        return (head - tail + Capacity) & (Capacity - 1);
    }

    // Get capacity
    constexpr size_t capacity() const {
        return Capacity - 1;  // One slot is always empty
    }

    // Clear the buffer
    void clear() {
        tail_.store(head_.load(std::memory_order_relaxed), std::memory_order_release);
    }

private:
    // Align head and tail to separate cache lines to avoid false sharing
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;
    alignas(CACHE_LINE_SIZE) std::array<T, Capacity> buffer_;
};

// Multi-producer, single-consumer variant (uses CAS for thread-safe push)
template <typename T, size_t Capacity>
class MPSCRingBuffer {
    static_assert(Capacity > 0, "Capacity must be greater than 0");
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    MPSCRingBuffer() : head_(0), tail_(0) {}

    // Thread-safe push (multiple producers)
    bool try_push(const T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);

        while (true) {
            const size_t next_head = (current_head + 1) & (Capacity - 1);

            // Check if full
            if (next_head == tail_.load(std::memory_order_acquire)) {
                return false;
            }

            // Try to claim the slot
            if (head_.compare_exchange_weak(current_head, next_head,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
                buffer_[current_head] = item;
                return true;
            }
            // CAS failed, current_head updated, retry
        }
    }

    // Single consumer pop
    std::optional<T> try_pop() {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);

        if (current_tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }

        T item = buffer_[current_tail];
        tail_.store((current_tail + 1) & (Capacity - 1), std::memory_order_release);
        return item;
    }

    bool empty() const {
        return tail_.load(std::memory_order_relaxed) ==
               head_.load(std::memory_order_relaxed);
    }

    size_t size() const {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_relaxed);
        return (head - tail + Capacity) & (Capacity - 1);
    }

private:
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;
    alignas(CACHE_LINE_SIZE) std::array<T, Capacity> buffer_;
};

}  // namespace argus
