#pragma once

// ============================================================================
// ArgusTrader Market Data Types
// ============================================================================

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace argus {
namespace market {

// Maximum symbol length (including null terminator)
constexpr size_t MAX_SYMBOL_LEN = 8;

// ============================================================================
// Trade (Tick) Data
// ============================================================================

struct alignas(64) Trade {
    uint64_t timestamp_ns;          // Nanoseconds since epoch
    char symbol[MAX_SYMBOL_LEN];    // Stock symbol
    double price;                   // Trade price
    uint32_t size;                  // Trade size (shares)
    char exchange;                  // Exchange code
    uint8_t conditions[4];          // Trade conditions
    uint8_t _padding[3];            // Padding for alignment

    Trade() : timestamp_ns(0), price(0.0), size(0), exchange(' ') {
        std::memset(symbol, 0, MAX_SYMBOL_LEN);
        std::memset(conditions, 0, 4);
    }

    void set_symbol(std::string_view sym) {
        size_t len = std::min(sym.size(), MAX_SYMBOL_LEN - 1);
        std::memcpy(symbol, sym.data(), len);
        symbol[len] = '\0';
    }

    std::string_view get_symbol() const {
        return std::string_view(symbol);
    }
};

static_assert(sizeof(Trade) == 64, "Trade must be 64 bytes (cache line aligned)");

// ============================================================================
// Quote Data
// ============================================================================

struct alignas(64) Quote {
    uint64_t timestamp_ns;
    char symbol[MAX_SYMBOL_LEN];
    double bid_price;
    double ask_price;
    uint32_t bid_size;
    uint32_t ask_size;
    char bid_exchange;
    char ask_exchange;
    uint8_t _padding[6];

    Quote() : timestamp_ns(0), bid_price(0.0), ask_price(0.0),
              bid_size(0), ask_size(0), bid_exchange(' '), ask_exchange(' ') {
        std::memset(symbol, 0, MAX_SYMBOL_LEN);
    }

    void set_symbol(std::string_view sym) {
        size_t len = std::min(sym.size(), MAX_SYMBOL_LEN - 1);
        std::memcpy(symbol, sym.data(), len);
        symbol[len] = '\0';
    }

    double spread() const {
        return ask_price - bid_price;
    }

    double mid_price() const {
        return (bid_price + ask_price) / 2.0;
    }
};

static_assert(sizeof(Quote) == 64, "Quote must be 64 bytes (cache line aligned)");

// ============================================================================
// OHLCV Bar (Aggregate)
// ============================================================================

struct Bar {
    uint64_t timestamp_ns;          // Bar start time
    char symbol[MAX_SYMBOL_LEN];
    double open;
    double high;
    double low;
    double close;
    uint64_t volume;
    double vwap;                    // Volume-weighted average price
    uint32_t trade_count;
    uint32_t bar_seconds;           // Bar duration (60 for 1-min, 300 for 5-min)

    Bar() : timestamp_ns(0), open(0), high(0), low(0), close(0),
            volume(0), vwap(0), trade_count(0), bar_seconds(60) {
        std::memset(symbol, 0, MAX_SYMBOL_LEN);
    }

    void set_symbol(std::string_view sym) {
        size_t len = std::min(sym.size(), MAX_SYMBOL_LEN - 1);
        std::memcpy(symbol, sym.data(), len);
        symbol[len] = '\0';
    }

    // Update bar with new trade
    void update(double price, uint32_t size) {
        if (trade_count == 0) {
            open = high = low = close = price;
        } else {
            high = std::max(high, price);
            low = std::min(low, price);
            close = price;
        }

        // Update VWAP
        double total_value = vwap * volume + price * size;
        volume += size;
        vwap = total_value / volume;
        trade_count++;
    }

    double range() const {
        return high - low;
    }

    double body() const {
        return close - open;
    }

    bool is_bullish() const {
        return close > open;
    }
};

// ============================================================================
// Polygon Message Types
// ============================================================================

enum class PolygonMsgType : uint8_t {
    UNKNOWN = 0,
    TRADE = 'T',
    QUOTE = 'Q',
    AGGREGATE_MINUTE = 'A',
    AGGREGATE_SECOND = 'a',
    STATUS = 's',
    ERROR = 'e'
};

// ============================================================================
// Market Status
// ============================================================================

enum class MarketStatus : uint8_t {
    UNKNOWN = 0,
    PRE_MARKET,
    OPEN,
    CLOSED,
    AFTER_HOURS,
    HALTED
};

inline std::string_view market_status_str(MarketStatus status) {
    switch (status) {
        case MarketStatus::PRE_MARKET:   return "PRE_MARKET";
        case MarketStatus::OPEN:         return "OPEN";
        case MarketStatus::CLOSED:       return "CLOSED";
        case MarketStatus::AFTER_HOURS:  return "AFTER_HOURS";
        case MarketStatus::HALTED:       return "HALTED";
        default:                         return "UNKNOWN";
    }
}

}  // namespace market
}  // namespace argus
