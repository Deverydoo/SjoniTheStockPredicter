// ============================================================================
// ArgusTrader - Main Entry Point
// ============================================================================

#include "utils/config.hpp"
#include "utils/logging.hpp"
#include "utils/time_utils.hpp"
#include "market_data/types.hpp"
#include "market_data/polygon_client.hpp"

#ifdef ARGUS_ONNX_ENABLED
#include "ml/onnx_predictor.hpp"
#endif

#ifdef ARGUS_ZMQ_ENABLED
#include "ipc/zmq_publisher.hpp"
#endif

#include <csignal>
#include <iostream>
#include <memory>
#include <thread>
#include <atomic>

namespace {
    std::atomic<bool> g_shutdown_requested{false};
}

void signal_handler(int signal) {
    spdlog::info("Received signal {}, initiating shutdown...", signal);
    g_shutdown_requested = true;
}

void print_banner() {
    std::cout << R"(
    ___                       ______               __
   /   |  _________ ___  ____/_  __/________ _____/ /__  _____
  / /| | / ___/ __ `/ / / / __/ / / ___/ __ `/ __  / _ \/ ___/
 / ___ |/ /  / /_/ / /_/ / /   / / /  / /_/ / /_/ /  __/ /
/_/  |_/_/   \__, /\__,_/_/   /_/_/   \__,_/\__,_/\___/_/
            /____/

    High-Performance NASDAQ Trading Engine
    Version 1.0.0

)" << std::endl;
}

int main(int argc, char* argv[]) {
    print_banner();

    // Parse command line
    std::string log_level = "INFO";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug") {
            log_level = "DEBUG";
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: argus_engine [options]\n"
                      << "Options:\n"
                      << "  --debug     Enable debug logging\n"
                      << "  -h, --help  Show this help\n";
            return 0;
        }
    }

    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Initialize logging
    argus::logging::init("./logs", log_level);

    spdlog::info("Starting ArgusTrader engine...");

    try {
        // Load config from environment
        auto config = argus::Config::from_env();

        // Validate required keys
        if (config.market_data.polygon.api_key.empty()) {
            spdlog::error("POLYGON_API_KEY environment variable not set");
            spdlog::info("Set it with: export POLYGON_API_KEY=your_key_here");
            return 1;
        }

        spdlog::info("Configuration loaded successfully");
        spdlog::info("Polygon API key: {}...", config.market_data.polygon.api_key.substr(0, 8));

#ifdef ARGUS_ZMQ_ENABLED
        // Create ZMQ publisher
        auto zmq_pub = std::make_shared<argus::ipc::ZMQPublisher>("tcp://*:5555");
        if (zmq_pub->start()) {
            spdlog::info("ZMQ publisher started on tcp://*:5555");
        } else {
            spdlog::warn("Failed to start ZMQ publisher");
        }
#endif

#ifdef ARGUS_ONNX_ENABLED
        // Create ML predictor
        auto predictor = std::make_shared<argus::ml::OnnxPredictor>();
        std::string model_dir = "D:/Vibe_Projects/The Trader/training/models";
        if (predictor->initialize(
                model_dir + "/model.onnx",
                model_dir + "/config.json",
                model_dir + "/norm_stats.json")) {
            spdlog::info("ML predictor initialized successfully");
        } else {
            spdlog::warn("Failed to initialize ML predictor");
        }

        // Per-symbol predictors for multi-symbol trading
        std::unordered_map<std::string, std::shared_ptr<argus::ml::OnnxPredictor>> symbol_predictors;
#endif

        // Create Polygon client
        auto polygon = std::make_shared<argus::market::PolygonClient>(config.market_data.polygon);

        // Setup callbacks for debugging
        polygon->on_trade([
#ifdef ARGUS_ZMQ_ENABLED
            zmq_pub
#endif
        ](const argus::market::Trade& trade) {
            spdlog::debug("TRADE: {} @ ${:.2f} x {} @ {}",
                         trade.get_symbol(),
                         trade.price,
                         trade.size,
                         argus::time::format_readable(argus::time::from_ns(trade.timestamp_ns)));

#ifdef ARGUS_ZMQ_ENABLED
            zmq_pub->publish_trade(trade);
#endif
        });

        polygon->on_bar([
#ifdef ARGUS_ONNX_ENABLED
            predictor,
            &symbol_predictors
#endif
#if defined(ARGUS_ZMQ_ENABLED) && defined(ARGUS_ONNX_ENABLED)
            ,
#endif
#ifdef ARGUS_ZMQ_ENABLED
            zmq_pub
#endif
        ](const argus::market::Bar& bar) {
            spdlog::info("BAR: {} O:{:.2f} H:{:.2f} L:{:.2f} C:{:.2f} V:{}",
                        bar.symbol,
                        bar.open, bar.high, bar.low, bar.close,
                        bar.volume);

#ifdef ARGUS_ZMQ_ENABLED
            zmq_pub->publish_bar(bar);
#endif

#ifdef ARGUS_ONNX_ENABLED
            // Add bar to predictor and check for signals
            argus::ml::Bar ml_bar{bar.open, bar.high, bar.low, bar.close,
                                   static_cast<double>(bar.volume), static_cast<int64_t>(bar.timestamp_ns)};
            predictor->add_bar(ml_bar);

            if (predictor->can_predict()) {
                auto prediction = predictor->predict();
                if (prediction.direction != 0) {
                    spdlog::info("SIGNAL [{}]: {} (return: {:.2f}%, confidence: {:.1f}%)",
                               bar.symbol,
                               prediction.signal,
                               prediction.predicted_return * 100,
                               prediction.confidence * 100);

#ifdef ARGUS_ZMQ_ENABLED
                    zmq_pub->publish_signal(bar.symbol, prediction.signal,
                                           prediction.confidence, "ML");
#endif
                }
            }
#endif
        });

        // Start the client
        spdlog::info("Connecting to Polygon.io...");
        polygon->start();

        // Main loop
        spdlog::info("Engine running. Press Ctrl+C to stop.");

        uint64_t last_stats_time = argus::time::now_ms();
        constexpr uint64_t STATS_INTERVAL_MS = 10000;  // Print stats every 10 seconds

        while (!g_shutdown_requested) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Print stats periodically
            uint64_t now = argus::time::now_ms();
            if (now - last_stats_time >= STATS_INTERVAL_MS) {
                spdlog::info("Stats: trades={}, quotes={}, bars={}, connected={}",
                           polygon->trades_received(),
                           polygon->quotes_received(),
                           polygon->bars_received(),
                           polygon->is_connected() ? "yes" : "no");
                last_stats_time = now;
            }
        }

        // Shutdown
        spdlog::info("Shutting down...");
        polygon->stop();

        spdlog::info("ArgusTrader stopped gracefully");

    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return 1;
    }

    argus::logging::shutdown();
    return 0;
}
