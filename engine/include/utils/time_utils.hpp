#pragma once

// ============================================================================
// ArgusTrader Time Utilities
// ============================================================================

#include <chrono>
#include <cstdint>
#include <string>
#include <iomanip>
#include <sstream>

namespace argus {
namespace time {

using Clock = std::chrono::system_clock;
using Timestamp = Clock::time_point;
using Nanoseconds = std::chrono::nanoseconds;
using Microseconds = std::chrono::microseconds;
using Milliseconds = std::chrono::milliseconds;
using Seconds = std::chrono::seconds;

// Get current time in nanoseconds since epoch
inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<Nanoseconds>(
            Clock::now().time_since_epoch()
        ).count()
    );
}

// Get current time in milliseconds since epoch
inline uint64_t now_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<Milliseconds>(
            Clock::now().time_since_epoch()
        ).count()
    );
}

// Get current time in microseconds since epoch
inline uint64_t now_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<Microseconds>(
            Clock::now().time_since_epoch()
        ).count()
    );
}

// Convert nanoseconds to Timestamp
inline Timestamp from_ns(uint64_t ns) {
    // Convert nanoseconds to the system_clock's duration type
    auto duration = std::chrono::duration_cast<Clock::duration>(Nanoseconds(ns));
    return Timestamp(duration);
}

// Convert milliseconds to Timestamp
inline Timestamp from_ms(uint64_t ms) {
    // Convert milliseconds to the system_clock's duration type
    auto duration = std::chrono::duration_cast<Clock::duration>(Milliseconds(ms));
    return Timestamp(duration);
}

// Convert Timestamp to nanoseconds
inline uint64_t to_ns(Timestamp ts) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<Nanoseconds>(
            ts.time_since_epoch()
        ).count()
    );
}

// Format timestamp as ISO 8601 string
inline std::string format_iso(Timestamp ts) {
    auto time_t = Clock::to_time_t(ts);
    auto ms = std::chrono::duration_cast<Milliseconds>(
        ts.time_since_epoch()
    ).count() % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms << 'Z';
    return oss.str();
}

// Format timestamp as readable string
inline std::string format_readable(Timestamp ts) {
    auto time_t = Clock::to_time_t(ts);
    auto ms = std::chrono::duration_cast<Milliseconds>(
        ts.time_since_epoch()
    ).count() % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms;
    return oss.str();
}

// Parse polygon timestamp (nanoseconds since epoch)
inline Timestamp parse_polygon_timestamp(uint64_t polygon_ns) {
    return from_ns(polygon_ns);
}

// Check if market is open (simplified - US Eastern, 9:30 AM - 4:00 PM)
inline bool is_market_hours() {
    auto now = Clock::now();
    auto time_t = Clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);

    // Skip weekends
    if (tm.tm_wday == 0 || tm.tm_wday == 6) {
        return false;
    }

    // Convert to minutes since midnight
    int minutes = tm.tm_hour * 60 + tm.tm_min;

    // Market hours: 9:30 AM (570 min) to 4:00 PM (960 min) Eastern
    // Note: This is simplified, doesn't handle timezone properly
    return minutes >= 570 && minutes < 960;
}

// High-resolution timer for latency measurement
class ScopedTimer {
public:
    using HighResClock = std::chrono::high_resolution_clock;

    ScopedTimer() : start_(HighResClock::now()) {}

    // Get elapsed time in microseconds
    double elapsed_us() const {
        auto end = HighResClock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

    // Get elapsed time in milliseconds
    double elapsed_ms() const {
        return elapsed_us() / 1000.0;
    }

    // Reset timer
    void reset() {
        start_ = HighResClock::now();
    }

private:
    HighResClock::time_point start_;
};

}  // namespace time
}  // namespace argus
