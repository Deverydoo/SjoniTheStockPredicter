#pragma once

// ============================================================================
// ArgusTrader ZeroMQ Publisher
// ============================================================================
// Publishes market data to Python dashboard via ZeroMQ PUB/SUB.
// Compile with -DARGUS_ZMQ_ENABLED and link against libzmq/cppzmq.
// ============================================================================

#include "market_data/types.hpp"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <string>
#include <memory>

#ifdef ARGUS_ZMQ_ENABLED
#include <zmq.hpp>
#endif

namespace argus {
namespace ipc {

using json = nlohmann::json;

class ZMQPublisher {
public:
    explicit ZMQPublisher(const std::string& address = "tcp://*:5555")
        : address_(address)
        , running_(false)
        , messages_sent_(0) {
    }

    ~ZMQPublisher() {
        stop();
    }

    // Non-copyable
    ZMQPublisher(const ZMQPublisher&) = delete;
    ZMQPublisher& operator=(const ZMQPublisher&) = delete;

    bool start() {
#ifdef ARGUS_ZMQ_ENABLED
        try {
            context_ = std::make_unique<zmq::context_t>(1);
            socket_ = std::make_unique<zmq::socket_t>(*context_, zmq::socket_type::pub);

            // Set high water mark to prevent message buildup
            int hwm = 10000;
            socket_->set(zmq::sockopt::sndhwm, hwm);

            socket_->bind(address_);
            running_ = true;

            spdlog::info("ZMQ Publisher started on {}", address_);
            return true;

        } catch (const zmq::error_t& e) {
            spdlog::error("Failed to start ZMQ Publisher: {}", e.what());
            return false;
        }
#else
        spdlog::warn("ZMQ support not compiled in (ARGUS_ZMQ_ENABLED not defined)");
        return false;
#endif
    }

    void stop() {
#ifdef ARGUS_ZMQ_ENABLED
        running_ = false;
        if (socket_) {
            socket_->close();
            socket_.reset();
        }
        if (context_) {
            context_->close();
            context_.reset();
        }
        spdlog::info("ZMQ Publisher stopped. Messages sent: {}", messages_sent_.load());
#endif
    }

    bool is_running() const { return running_; }
    uint64_t messages_sent() const { return messages_sent_; }

    // Publish a trade
    void publish_trade(const market::Trade& trade) {
#ifdef ARGUS_ZMQ_ENABLED
        if (!running_) return;

        json msg = {
            {"symbol", std::string(trade.symbol)},
            {"timestamp", trade.timestamp_ns},
            {"price", trade.price},
            {"size", trade.size},
            {"exchange", std::string(1, trade.exchange)}
        };

        send_message("TRADE", msg);
#else
        (void)trade;
#endif
    }

    // Publish a quote
    void publish_quote(const market::Quote& quote) {
#ifdef ARGUS_ZMQ_ENABLED
        if (!running_) return;

        json msg = {
            {"symbol", std::string(quote.symbol)},
            {"timestamp", quote.timestamp_ns},
            {"bid_price", quote.bid_price},
            {"ask_price", quote.ask_price},
            {"bid_size", quote.bid_size},
            {"ask_size", quote.ask_size}
        };

        send_message("QUOTE", msg);
#else
        (void)quote;
#endif
    }

    // Publish a bar
    void publish_bar(const market::Bar& bar) {
#ifdef ARGUS_ZMQ_ENABLED
        if (!running_) return;

        json msg = {
            {"symbol", std::string(bar.symbol)},
            {"timestamp", bar.timestamp_ns},
            {"open", bar.open},
            {"high", bar.high},
            {"low", bar.low},
            {"close", bar.close},
            {"volume", bar.volume},
            {"vwap", bar.vwap}
        };

        send_message("BAR", msg);
#else
        (void)bar;
#endif
    }

    // Publish a trading signal
    void publish_signal(const std::string& symbol, const std::string& signal_type,
                        double strength, const std::string& source) {
#ifdef ARGUS_ZMQ_ENABLED
        if (!running_) return;

        json msg = {
            {"symbol", symbol},
            {"timestamp", std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"signal_type", signal_type},
            {"strength", strength},
            {"source", source}
        };

        send_message("SIGNAL", msg);
#else
        (void)symbol; (void)signal_type; (void)strength; (void)source;
#endif
    }

private:
    void send_message(const std::string& topic, const json& data) {
#ifdef ARGUS_ZMQ_ENABLED
        try {
            std::string json_str = data.dump();

            // Send topic frame
            zmq::message_t topic_msg(topic.data(), topic.size());
            socket_->send(topic_msg, zmq::send_flags::sndmore);

            // Send data frame
            zmq::message_t data_msg(json_str.data(), json_str.size());
            socket_->send(data_msg, zmq::send_flags::none);

            messages_sent_++;

        } catch (const zmq::error_t& e) {
            spdlog::warn("ZMQ send error: {}", e.what());
        }
#else
        (void)topic; (void)data;
#endif
    }

private:
    std::string address_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> messages_sent_;

#ifdef ARGUS_ZMQ_ENABLED
    std::unique_ptr<zmq::context_t> context_;
    std::unique_ptr<zmq::socket_t> socket_;
#endif
};

}  // namespace ipc
}  // namespace argus
