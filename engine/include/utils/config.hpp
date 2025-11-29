#pragma once

// ============================================================================
// ArgusTrader Configuration
// ============================================================================

#include <string>
#include <vector>
#include <optional>
#include <fstream>
#include <stdexcept>

namespace argus {

struct PolygonConfig {
    std::string api_key;
    std::string websocket_url = "wss://socket.polygon.io/stocks";
    std::string rest_url = "https://api.polygon.io";
    std::vector<std::string> subscriptions = {"T.*", "AM.*"};  // Trades, Minute aggs
    size_t buffer_size = 1'000'000;
    int reconnect_delay_ms = 1000;
    int max_reconnect_attempts = 10;
};

struct AlpacaConfig {
    std::string api_key;
    std::string secret_key;
    bool paper = true;
    std::string data_url = "https://data.alpaca.markets";
    std::string trading_url = "https://paper-api.alpaca.markets";
};

struct MarketDataConfig {
    PolygonConfig polygon;
    AlpacaConfig alpaca;
};

struct DatabaseConfig {
    std::string host = "localhost";
    int port = 5432;
    std::string user = "argus";
    std::string password;
    std::string database = "argus_trader";
    int pool_size = 10;
};

struct RedisConfig {
    std::string host = "localhost";
    int port = 6379;
    std::string password;
    int database = 0;
};

struct ZmqConfig {
    std::string pub_endpoint = "tcp://*:5555";
    std::string sub_endpoint = "tcp://*:5556";
    int high_water_mark = 10000;
};

struct LoggingConfig {
    std::string level = "INFO";
    std::string log_dir = "./logs";
    size_t max_file_size_mb = 100;
    int max_files = 10;
};

struct Config {
    MarketDataConfig market_data;
    DatabaseConfig database;
    RedisConfig redis;
    ZmqConfig zmq;
    LoggingConfig logging;

    // Load from environment variables (simple approach for now)
    static Config from_env() {
        Config config;

        // Polygon
        if (const char* val = std::getenv("POLYGON_API_KEY")) {
            config.market_data.polygon.api_key = val;
        }

        // Alpaca
        if (const char* val = std::getenv("ALPACA_API_KEY")) {
            config.market_data.alpaca.api_key = val;
        }
        if (const char* val = std::getenv("ALPACA_SECRET_KEY")) {
            config.market_data.alpaca.secret_key = val;
        }

        // Database
        if (const char* val = std::getenv("DB_HOST")) {
            config.database.host = val;
        }
        if (const char* val = std::getenv("DB_PORT")) {
            config.database.port = std::stoi(val);
        }
        if (const char* val = std::getenv("DB_USER")) {
            config.database.user = val;
        }
        if (const char* val = std::getenv("DB_PASSWORD")) {
            config.database.password = val;
        }
        if (const char* val = std::getenv("DB_NAME")) {
            config.database.database = val;
        }

        // Redis
        if (const char* val = std::getenv("REDIS_HOST")) {
            config.redis.host = val;
        }
        if (const char* val = std::getenv("REDIS_PORT")) {
            config.redis.port = std::stoi(val);
        }

        // Logging
        if (const char* val = std::getenv("LOG_LEVEL")) {
            config.logging.level = val;
        }
        if (const char* val = std::getenv("LOG_DIR")) {
            config.logging.log_dir = val;
        }

        return config;
    }

    void validate() const {
        if (market_data.polygon.api_key.empty()) {
            throw std::runtime_error("POLYGON_API_KEY not set");
        }
    }
};

}  // namespace argus
