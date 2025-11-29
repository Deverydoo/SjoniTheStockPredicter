#pragma once

// ============================================================================
// ArgusTrader Polygon.io WebSocket Client
// ============================================================================

#include "market_data/types.hpp"
#include "utils/ring_buffer.hpp"
#include "utils/config.hpp"

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace argus {
namespace market {

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = net::ssl;
using tcp = net::ip::tcp;
using json = nlohmann::json;

// Callback types
using TradeCallback = std::function<void(const Trade&)>;
using QuoteCallback = std::function<void(const Quote&)>;
using BarCallback = std::function<void(const Bar&)>;

class PolygonClient : public std::enable_shared_from_this<PolygonClient> {
public:
    // Ring buffer sizes (must be power of 2)
    static constexpr size_t TRADE_BUFFER_SIZE = 1 << 20;  // ~1M trades
    static constexpr size_t QUOTE_BUFFER_SIZE = 1 << 20;
    static constexpr size_t BAR_BUFFER_SIZE = 1 << 16;    // ~65K bars

    explicit PolygonClient(const PolygonConfig& config)
        : config_(config)
        , ssl_ctx_(ssl::context::tlsv12_client)
        , resolver_(ioc_)
        , ws_(ioc_, ssl_ctx_)
        , connected_(false)
        , running_(false)
        , reconnect_attempts_(0) {

        // Configure SSL
        ssl_ctx_.set_default_verify_paths();
        ssl_ctx_.set_verify_mode(ssl::verify_peer);
    }

    ~PolygonClient() {
        stop();
    }

    // Non-copyable
    PolygonClient(const PolygonClient&) = delete;
    PolygonClient& operator=(const PolygonClient&) = delete;

    // Set callbacks
    void on_trade(TradeCallback cb) { trade_callback_ = std::move(cb); }
    void on_quote(QuoteCallback cb) { quote_callback_ = std::move(cb); }
    void on_bar(BarCallback cb) { bar_callback_ = std::move(cb); }

    // Start connection
    void start() {
        if (running_.exchange(true)) {
            return;  // Already running
        }

        spdlog::info("Starting Polygon WebSocket client");

        // Run IO context in separate thread
        io_thread_ = std::thread([this]() {
            while (running_) {
                try {
                    connect();
                    ioc_.run();
                } catch (const std::exception& e) {
                    spdlog::error("Polygon client error: {}", e.what());
                    handle_disconnect();
                }

                if (running_) {
                    // Wait before reconnecting
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(config_.reconnect_delay_ms)
                    );
                    ioc_.restart();
                }
            }
        });
    }

    // Stop connection
    void stop() {
        if (!running_.exchange(false)) {
            return;  // Not running
        }

        spdlog::info("Stopping Polygon WebSocket client");

        // Close WebSocket gracefully
        if (connected_) {
            beast::error_code ec;
            ws_.close(websocket::close_code::normal, ec);
        }

        ioc_.stop();

        if (io_thread_.joinable()) {
            io_thread_.join();
        }

        connected_ = false;
    }

    // Check connection status
    bool is_connected() const { return connected_; }

    // Get buffer references for direct access
    RingBuffer<Trade, TRADE_BUFFER_SIZE>& trade_buffer() { return trade_buffer_; }
    RingBuffer<Quote, QUOTE_BUFFER_SIZE>& quote_buffer() { return quote_buffer_; }
    RingBuffer<Bar, BAR_BUFFER_SIZE>& bar_buffer() { return bar_buffer_; }

    // Statistics
    uint64_t trades_received() const { return trades_received_; }
    uint64_t quotes_received() const { return quotes_received_; }
    uint64_t bars_received() const { return bars_received_; }

private:
    void connect() {
        spdlog::info("Connecting to Polygon WebSocket...");

        // Parse URL
        std::string host = "socket.polygon.io";
        std::string port = "443";
        std::string target = "/stocks";

        // Resolve
        auto const results = resolver_.resolve(host, port);

        // Connect TCP
        auto ep = net::connect(beast::get_lowest_layer(ws_), results);

        // Set SNI hostname
        if (!SSL_set_tlsext_host_name(ws_.next_layer().native_handle(), host.c_str())) {
            throw beast::system_error(
                beast::error_code(static_cast<int>(::ERR_get_error()),
                                  net::error::get_ssl_category()),
                "Failed to set SNI hostname");
        }

        // SSL handshake
        ws_.next_layer().handshake(ssl::stream_base::client);

        // Set WebSocket options
        ws_.set_option(websocket::stream_base::decorator(
            [](websocket::request_type& req) {
                req.set(beast::http::field::user_agent, "ArgusTrader/1.0");
            }));

        // WebSocket handshake
        ws_.handshake(host, target);

        connected_ = true;
        reconnect_attempts_ = 0;
        spdlog::info("Connected to Polygon WebSocket");

        // Authenticate
        authenticate();

        // Subscribe to channels
        subscribe();

        // Start reading
        read_loop();
    }

    void authenticate() {
        json auth_msg = {
            {"action", "auth"},
            {"params", config_.api_key}
        };

        ws_.write(net::buffer(auth_msg.dump()));
        spdlog::debug("Sent authentication request");
    }

    void subscribe() {
        for (const auto& channel : config_.subscriptions) {
            json sub_msg = {
                {"action", "subscribe"},
                {"params", channel}
            };

            ws_.write(net::buffer(sub_msg.dump()));
            spdlog::info("Subscribed to: {}", channel);
        }
    }

    void read_loop() {
        while (running_ && connected_) {
            try {
                beast::flat_buffer buffer;
                ws_.read(buffer);

                std::string msg = beast::buffers_to_string(buffer.data());
                process_message(msg);

            } catch (const beast::system_error& e) {
                if (e.code() != websocket::error::closed) {
                    throw;
                }
                break;
            }
        }
    }

    void process_message(const std::string& msg) {
        try {
            auto data = json::parse(msg);

            // Polygon sends arrays of messages
            if (!data.is_array()) {
                data = json::array({data});
            }

            for (const auto& item : data) {
                if (!item.contains("ev")) continue;

                std::string event = item["ev"].get<std::string>();

                if (event == "T") {
                    process_trade(item);
                } else if (event == "Q") {
                    process_quote(item);
                } else if (event == "AM" || event == "A") {
                    process_bar(item);
                } else if (event == "status") {
                    process_status(item);
                }
            }

        } catch (const json::exception& e) {
            spdlog::warn("JSON parse error: {}", e.what());
        }
    }

    void process_trade(const json& data) {
        Trade trade;

        // Parse fields
        if (data.contains("sym")) {
            trade.set_symbol(data["sym"].get<std::string>());
        }
        if (data.contains("t")) {
            // Polygon timestamp is in nanoseconds
            trade.timestamp_ns = data["t"].get<uint64_t>();
        }
        if (data.contains("p")) {
            trade.price = data["p"].get<double>();
        }
        if (data.contains("s")) {
            trade.size = data["s"].get<uint32_t>();
        }
        if (data.contains("x")) {
            trade.exchange = static_cast<char>(data["x"].get<int>());
        }
        if (data.contains("c") && data["c"].is_array()) {
            auto& conditions = data["c"];
            for (size_t i = 0; i < std::min(conditions.size(), size_t(4)); ++i) {
                trade.conditions[i] = static_cast<uint8_t>(conditions[i].get<int>());
            }
        }

        // Push to buffer
        trade_buffer_.push_overwrite(trade);
        trades_received_++;

        // Call callback if set
        if (trade_callback_) {
            trade_callback_(trade);
        }
    }

    void process_quote(const json& data) {
        Quote quote;

        if (data.contains("sym")) {
            quote.set_symbol(data["sym"].get<std::string>());
        }
        if (data.contains("t")) {
            quote.timestamp_ns = data["t"].get<uint64_t>();
        }
        if (data.contains("bp")) {
            quote.bid_price = data["bp"].get<double>();
        }
        if (data.contains("ap")) {
            quote.ask_price = data["ap"].get<double>();
        }
        if (data.contains("bs")) {
            quote.bid_size = data["bs"].get<uint32_t>();
        }
        if (data.contains("as")) {
            quote.ask_size = data["as"].get<uint32_t>();
        }

        quote_buffer_.push_overwrite(quote);
        quotes_received_++;

        if (quote_callback_) {
            quote_callback_(quote);
        }
    }

    void process_bar(const json& data) {
        Bar bar;

        if (data.contains("sym")) {
            bar.set_symbol(data["sym"].get<std::string>());
        }
        if (data.contains("s")) {
            // Start timestamp in milliseconds, convert to ns
            bar.timestamp_ns = data["s"].get<uint64_t>() * 1'000'000;
        }
        if (data.contains("o")) {
            bar.open = data["o"].get<double>();
        }
        if (data.contains("h")) {
            bar.high = data["h"].get<double>();
        }
        if (data.contains("l")) {
            bar.low = data["l"].get<double>();
        }
        if (data.contains("c")) {
            bar.close = data["c"].get<double>();
        }
        if (data.contains("v")) {
            bar.volume = data["v"].get<uint64_t>();
        }
        if (data.contains("vw")) {
            bar.vwap = data["vw"].get<double>();
        }
        if (data.contains("n")) {
            bar.trade_count = data["n"].get<uint32_t>();
        }

        bar_buffer_.push_overwrite(bar);
        bars_received_++;

        if (bar_callback_) {
            bar_callback_(bar);
        }
    }

    void process_status(const json& data) {
        if (data.contains("message")) {
            spdlog::info("Polygon status: {}", data["message"].get<std::string>());
        }
    }

    void handle_disconnect() {
        connected_ = false;
        reconnect_attempts_++;

        if (reconnect_attempts_ >= config_.max_reconnect_attempts) {
            spdlog::error("Max reconnect attempts reached, giving up");
            running_ = false;
        } else {
            spdlog::warn("Disconnected, attempt {} of {}",
                        reconnect_attempts_, config_.max_reconnect_attempts);
        }
    }

private:
    PolygonConfig config_;

    // ASIO/Beast
    net::io_context ioc_;
    ssl::context ssl_ctx_;
    tcp::resolver resolver_;
    websocket::stream<beast::ssl_stream<tcp::socket>> ws_;

    // State
    std::atomic<bool> connected_;
    std::atomic<bool> running_;
    int reconnect_attempts_;

    // Thread
    std::thread io_thread_;

    // Callbacks
    TradeCallback trade_callback_;
    QuoteCallback quote_callback_;
    BarCallback bar_callback_;

    // Ring buffers
    RingBuffer<Trade, TRADE_BUFFER_SIZE> trade_buffer_;
    RingBuffer<Quote, QUOTE_BUFFER_SIZE> quote_buffer_;
    RingBuffer<Bar, BAR_BUFFER_SIZE> bar_buffer_;

    // Statistics
    std::atomic<uint64_t> trades_received_{0};
    std::atomic<uint64_t> quotes_received_{0};
    std::atomic<uint64_t> bars_received_{0};
};

}  // namespace market
}  // namespace argus
