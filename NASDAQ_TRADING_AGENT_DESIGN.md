# NASDAQ Trading Agent - Design Document

## Project Codename: **ArgusTrader**

**Version**: 1.0
**Date**: November 2025
**Authors**: Craig Giannelli, Claude Code

---

## Executive Summary

ArgusTrader is a high-performance NASDAQ monitoring and prediction system designed for algorithmic trading. The system combines real-time market data analysis, machine learning predictions, and news sentiment analysis to identify high-probability trading opportunities.

### Key Differentiators
- **C++ Core Engine**: Sub-millisecond inference latency
- **Multi-Signal Fusion**: Technical analysis + ML predictions + news sentiment
- **Dynamic Stock Selection**: Automated top-20 screening from full NASDAQ universe
- **Python Dash Interface**: Real-time visualization and control

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Pipeline](#2-data-pipeline)
3. [Stock Screening Engine](#3-stock-screening-engine)
4. [Prediction Engine (C++)](#4-prediction-engine-c)
5. [News Sentiment Analysis](#5-news-sentiment-analysis)
6. [Signal Fusion & Decision Engine](#6-signal-fusion--decision-engine)
7. [Python Dash Dashboard](#7-python-dash-dashboard)
8. [Infrastructure & Deployment](#8-infrastructure--deployment)
9. [Risk Management](#9-risk-management)
10. [Development Roadmap](#10-development-roadmap)
11. [Appendices](#11-appendices)

---

## 1. System Architecture

### 1.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARGUS TRADER SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ Market Data  │    │  News Feeds  │    │   SEC EDGAR  │                   │
│  │   Polygon    │    │  NewsAPI     │    │   Filings    │                   │
│  │   Alpaca     │    │  Benzinga    │    │              │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    DATA INGESTION LAYER (C++)                    │        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │        │
│  │  │ WebSocket   │  │  REST API   │  │   Message   │              │        │
│  │  │  Handler    │  │   Client    │  │    Queue    │              │        │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    PROCESSING LAYER (C++)                        │        │
│  │                                                                  │        │
│  │  ┌─────────────────┐    ┌─────────────────┐                     │        │
│  │  │  Stock Screener │    │ Sentiment Engine│                     │        │
│  │  │  (Top 20 Pick)  │    │  (FinBERT/LLM)  │                     │        │
│  │  └────────┬────────┘    └────────┬────────┘                     │        │
│  │           │                      │                               │        │
│  │           ▼                      ▼                               │        │
│  │  ┌─────────────────────────────────────────────────────┐        │        │
│  │  │              TensorRT Inference Engine               │        │        │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │        │        │
│  │  │  │  Price  │  │ Volume  │  │Sentiment│             │        │        │
│  │  │  │Predictor│  │Predictor│  │ Scorer  │             │        │        │
│  │  │  └─────────┘  └─────────┘  └─────────┘             │        │        │
│  │  └─────────────────────────────────────────────────────┘        │        │
│  │                              │                                   │        │
│  │                              ▼                                   │        │
│  │  ┌─────────────────────────────────────────────────────┐        │        │
│  │  │              Signal Fusion Engine                    │        │        │
│  │  │         (Weighted Multi-Factor Scoring)              │        │        │
│  │  └─────────────────────────────────────────────────────┘        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                │                                             │
│                                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    OUTPUT LAYER                                  │        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │        │
│  │  │   ZeroMQ    │  │  REST API   │  │  Database   │              │        │
│  │  │  Publisher  │  │   Server    │  │  (TimescaleDB)             │        │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────────┘              │        │
│  └─────────┼────────────────┼──────────────────────────────────────┘        │
│            │                │                                                │
│            ▼                ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                 PYTHON DASH DASHBOARD                            │        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │        │
│  │  │  Real-time  │  │  Portfolio  │  │   Alerts &  │              │        │
│  │  │   Charts    │  │   Manager   │  │   Signals   │              │        │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Core Engine** | C++20 | Zero-overhead abstractions, deterministic latency |
| **ML Inference** | TensorRT 8.x | Optimized GPU inference, INT8/FP16 support |
| **Sentiment Model** | FinBERT (ONNX) | Financial domain-specific NLP |
| **Message Queue** | ZeroMQ | Lock-free, microsecond latency |
| **Database** | TimescaleDB | Time-series optimized, SQL compatible |
| **Cache** | Redis | Sub-millisecond key-value lookups |
| **Dashboard** | Python Dash + Plotly | Rapid development, real-time updates |
| **IPC** | Shared Memory + ZeroMQ | C++ ↔ Python communication |

### 1.3 Latency Targets

| Operation | Target | Measurement Point |
|-----------|--------|-------------------|
| Market data ingestion | < 100 μs | Wire → Memory |
| Stock screening update | < 1 ms | Per screening cycle |
| Price prediction inference | < 500 μs | Input → Output |
| Sentiment scoring | < 5 ms | Text → Score |
| Signal fusion | < 100 μs | Signals → Decision |
| Dashboard update | < 100 ms | Engine → Browser |
| **End-to-end (data → signal)** | **< 10 ms** | Wire → Recommendation |

---

## 2. Data Pipeline

### 2.1 Market Data Sources

#### Primary: Polygon.io
```cpp
// WebSocket connection for real-time data
struct MarketDataConfig {
    std::string api_key;
    std::vector<std::string> subscriptions;  // "T.*" for all trades
    bool enable_aggregates = true;           // 1-second bars
    size_t buffer_size = 1'000'000;          // Pre-allocated buffer
};

// Data structure (cache-line aligned)
struct alignas(64) TickData {
    uint64_t timestamp_ns;      // Nanosecond precision
    char symbol[8];             // Null-terminated, fixed size
    float price;
    uint32_t volume;
    char exchange;              // Exchange code
    uint8_t conditions[4];      // Trade conditions
};
```

#### Secondary: Alpaca Markets (Paper Trading + Live)
- Commission-free trading API
- Real-time and historical data
- Built-in paper trading for backtesting

### 2.2 News Data Sources

| Source | Type | Latency | Cost |
|--------|------|---------|------|
| **Benzinga Pro** | Premium news wire | < 1 second | $$$$ |
| **NewsAPI** | Aggregated news | 1-15 minutes | $ |
| **Alpha Vantage** | News + sentiment | 1-5 minutes | $$ |
| **SEC EDGAR** | Filings, 8-K, 10-Q | Minutes | Free |
| **Twitter/X API** | Social sentiment | Real-time | $$$ |

### 2.3 Data Storage Schema

```sql
-- TimescaleDB hypertable for tick data
CREATE TABLE market_ticks (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    price       DOUBLE PRECISION,
    volume      BIGINT,
    exchange    CHAR(1),
    conditions  INTEGER[]
);
SELECT create_hypertable('market_ticks', 'time');
CREATE INDEX idx_symbol_time ON market_ticks (symbol, time DESC);

-- Aggregated OHLCV bars
CREATE TABLE ohlcv_1m (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    vwap        DOUBLE PRECISION,
    trade_count INTEGER
);
SELECT create_hypertable('ohlcv_1m', 'time');

-- News sentiment cache
CREATE TABLE news_sentiment (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    headline    TEXT,
    source      TEXT,
    sentiment   DOUBLE PRECISION,  -- -1.0 to 1.0
    confidence  DOUBLE PRECISION,  -- 0.0 to 1.0
    url         TEXT
);
SELECT create_hypertable('news_sentiment', 'time');

-- Stock screening results
CREATE TABLE screener_rankings (
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    rank            INTEGER,
    composite_score DOUBLE PRECISION,
    momentum_score  DOUBLE PRECISION,
    volume_score    DOUBLE PRECISION,
    sentiment_score DOUBLE PRECISION,
    prediction_score DOUBLE PRECISION
);
SELECT create_hypertable('screener_rankings', 'time');
```

---

## 3. Stock Screening Engine

### 3.1 Universe Definition

```cpp
// NASDAQ-100 + High-volume NASDAQ stocks
constexpr size_t MAX_UNIVERSE_SIZE = 500;
constexpr size_t TOP_N_SELECTION = 20;

struct UniverseFilter {
    float min_market_cap = 1e9f;        // $1B minimum
    float min_avg_volume = 500'000;      // 500K shares/day
    float min_price = 5.0f;              // Above $5 (no penny stocks)
    float max_spread_pct = 0.5f;         // Max 0.5% bid-ask spread
    bool nasdaq_only = true;
    bool exclude_adrs = true;
    bool exclude_etfs = false;           // Some ETFs like QQQ are useful
};
```

### 3.2 Multi-Factor Screening Algorithm

```cpp
class StockScreener {
public:
    struct ScreeningFactors {
        // Momentum factors (40% weight)
        float price_momentum_5d;     // 5-day price change
        float price_momentum_20d;    // 20-day price change
        float rsi_14;                // Relative Strength Index
        float macd_signal;           // MACD histogram

        // Volume factors (20% weight)
        float volume_surge;          // Current vs 20-day average
        float accumulation_dist;     // Accumulation/Distribution
        float obv_trend;             // On-Balance Volume trend

        // Volatility factors (15% weight)
        float atr_percentile;        // ATR vs historical
        float bollinger_position;    // Position within bands
        float implied_vol_rank;      // IV percentile (if options data)

        // Sentiment factors (25% weight)
        float news_sentiment_24h;    // Aggregated news sentiment
        float news_volume;           // Number of articles (attention)
        float social_sentiment;      // Twitter/StockTwits sentiment
        float analyst_revision;      // Recent estimate changes
    };

    struct ScoredStock {
        char symbol[8];
        float composite_score;       // Weighted combination
        float confidence;            // Model confidence
        ScreeningFactors factors;
        uint64_t last_update_ns;
    };

    // Main screening function - runs every minute
    std::array<ScoredStock, TOP_N_SELECTION> screen_top_stocks();

private:
    // Factor weights (tunable via config)
    static constexpr float MOMENTUM_WEIGHT = 0.40f;
    static constexpr float VOLUME_WEIGHT = 0.20f;
    static constexpr float VOLATILITY_WEIGHT = 0.15f;
    static constexpr float SENTIMENT_WEIGHT = 0.25f;

    // Z-score normalization for cross-stock comparison
    float normalize_factor(float value, const FactorStats& stats);

    // Composite score calculation
    float calculate_composite_score(const ScreeningFactors& factors);
};
```

### 3.3 Screening Frequency

| Timeframe | Action | Rationale |
|-----------|--------|-----------|
| **Every 1 minute** | Update factor scores | React to price movements |
| **Every 5 minutes** | Re-rank universe | Balance responsiveness vs stability |
| **Every 15 minutes** | Refresh news sentiment | News doesn't change that fast |
| **Daily (pre-market)** | Full universe scan | Add/remove stocks from universe |

---

## 4. Prediction Engine (C++)

### 4.1 Model Architecture

We adapt the Temporal Fusion Transformer (TFT) architecture for financial time series:

```cpp
// Model configuration
struct TFTPredictorConfig {
    // Input configuration
    size_t sequence_length = 60;        // 60 minutes of history
    size_t prediction_horizon = 12;     // Predict 12 steps ahead
    size_t num_features = 32;           // OHLCV + technicals + sentiment

    // Architecture
    size_t hidden_size = 128;
    size_t num_attention_heads = 4;
    size_t num_lstm_layers = 2;
    size_t num_static_features = 8;     // Sector, market cap bucket, etc.

    // Inference optimization
    bool use_fp16 = true;
    bool use_int8 = false;              // Requires calibration
    size_t max_batch_size = 32;         // Process multiple stocks together
};
```

### 4.2 TensorRT Integration

```cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>

class TensorRTPredictor {
public:
    TensorRTPredictor(const std::string& engine_path) {
        // Load serialized TensorRT engine
        std::ifstream file(engine_path, std::ios::binary);
        std::vector<char> engine_data(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );

        runtime_ = nvinfer1::createInferRuntime(logger_);
        engine_ = runtime_->deserializeCudaEngine(
            engine_data.data(), engine_data.size()
        );
        context_ = engine_->createExecutionContext();

        // Pre-allocate GPU memory
        allocate_buffers();
    }

    struct PredictionResult {
        std::array<float, 12> price_predictions;  // Point predictions
        std::array<float, 12> lower_bound;        // 10th percentile
        std::array<float, 12> upper_bound;        // 90th percentile
        float trend_probability;                   // P(up) vs P(down)
        float confidence;
    };

    // Batch prediction for multiple stocks
    std::vector<PredictionResult> predict(
        const std::vector<StockFeatures>& inputs
    ) {
        // Copy input to GPU (async)
        cudaMemcpyAsync(
            d_input_, inputs.data(),
            inputs.size() * sizeof(StockFeatures),
            cudaMemcpyHostToDevice, stream_
        );

        // Execute inference
        context_->enqueueV2(buffers_.data(), stream_, nullptr);

        // Copy output from GPU (async)
        std::vector<PredictionResult> results(inputs.size());
        cudaMemcpyAsync(
            results.data(), d_output_,
            results.size() * sizeof(PredictionResult),
            cudaMemcpyDeviceToHost, stream_
        );

        // Synchronize
        cudaStreamSynchronize(stream_);
        return results;
    }

private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;

    void* d_input_;
    void* d_output_;
    std::vector<void*> buffers_;

    Logger logger_;  // Custom TensorRT logger
};
```

### 4.3 Feature Engineering

```cpp
struct StockFeatures {
    // Time-series features (60 timesteps x N features)
    float ohlcv[60][5];              // Open, High, Low, Close, Volume
    float returns[60];               // Log returns
    float volatility[60];            // Rolling volatility

    // Technical indicators
    float sma_10[60];
    float sma_20[60];
    float ema_12[60];
    float ema_26[60];
    float rsi_14[60];
    float macd[60];
    float macd_signal[60];
    float bollinger_upper[60];
    float bollinger_lower[60];
    float atr_14[60];
    float obv[60];
    float vwap[60];

    // Sentiment features (aggregated per timestep)
    float news_sentiment[60];
    float news_volume[60];
    float social_sentiment[60];

    // Static features (per stock)
    float market_cap_bucket;         // Encoded: mega, large, mid, small
    float sector_encoding[11];       // One-hot GICS sectors
    float beta;                      // Market beta
    float avg_spread;                // Average bid-ask spread
};

class FeatureEngine {
public:
    StockFeatures compute_features(
        const std::string& symbol,
        const std::vector<OHLCVBar>& bars,
        const std::vector<NewsSentiment>& news
    );

private:
    // Technical indicator calculators
    TechnicalIndicators indicators_;

    // Normalization statistics (loaded from training)
    NormalizationStats norm_stats_;
};
```

### 4.4 Model Training Pipeline (Python)

Training remains in Python for flexibility, then export to TensorRT:

```python
# training/train_trading_model.py

import pytorch_forecasting as pf
import torch
from torch.utils.data import DataLoader

class TradingTFT(pf.TemporalFusionTransformer):
    """Custom TFT for trading predictions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom output head for trading signals
        self.trend_head = nn.Linear(self.hparams.hidden_size, 2)  # Up/Down

    def forward(self, x):
        # Standard TFT forward
        output = super().forward(x)

        # Additional trend classification
        hidden = self.get_hidden_state(x)
        trend_logits = self.trend_head(hidden)
        trend_probs = F.softmax(trend_logits, dim=-1)

        return {
            'prediction': output.prediction,
            'quantiles': output.quantiles,
            'trend_probability': trend_probs[:, 1]  # P(up)
        }

def export_to_onnx(model, sample_input, output_path):
    """Export trained model to ONNX format."""
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=['features', 'static'],
        output_names=['prediction', 'quantiles', 'trend_prob'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'prediction': {0: 'batch_size'}
        },
        opset_version=17
    )

def convert_onnx_to_tensorrt(onnx_path, engine_path, fp16=True):
    """Convert ONNX model to TensorRT engine."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    # Build config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build engine
    engine = builder.build_engine(network, config)

    # Serialize
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

---

## 5. News Sentiment Analysis

### 5.1 FinBERT Model

```cpp
// Sentiment analysis using FinBERT (financial domain BERT)
class SentimentAnalyzer {
public:
    struct SentimentResult {
        float score;           // -1.0 (bearish) to +1.0 (bullish)
        float confidence;      // 0.0 to 1.0
        std::string label;     // "bullish", "bearish", "neutral"
    };

    SentimentAnalyzer(const std::string& model_path) {
        // Load FinBERT TensorRT engine
        engine_ = load_tensorrt_engine(model_path);
        tokenizer_ = load_tokenizer("finbert-tokenizer.json");
    }

    SentimentResult analyze(const std::string& text) {
        // Tokenize
        auto tokens = tokenizer_.encode(text, MAX_SEQUENCE_LENGTH);

        // Run inference
        auto logits = engine_->infer(tokens);

        // Softmax to probabilities
        auto probs = softmax(logits);

        // Convert to score: bearish=-1, neutral=0, bullish=+1
        float score = probs[2] - probs[0];  // bullish - bearish
        float confidence = std::max({probs[0], probs[1], probs[2]});

        return {score, confidence, get_label(probs)};
    }

    // Batch processing for efficiency
    std::vector<SentimentResult> analyze_batch(
        const std::vector<std::string>& texts
    );

private:
    std::unique_ptr<TensorRTEngine> engine_;
    Tokenizer tokenizer_;

    static constexpr size_t MAX_SEQUENCE_LENGTH = 512;
};
```

### 5.2 News Aggregation Strategy

```cpp
struct NewsAggregator {
    struct AggregatedSentiment {
        float weighted_sentiment;    // Time-decayed weighted average
        float sentiment_momentum;    // Rate of change
        int article_count;           // Number of articles
        float attention_score;       // Unusual news volume indicator
        std::string top_headline;    // Most impactful headline
    };

    AggregatedSentiment aggregate_for_symbol(
        const std::string& symbol,
        std::chrono::hours lookback = std::chrono::hours(24)
    ) {
        auto articles = fetch_recent_articles(symbol, lookback);

        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;
        auto now = std::chrono::system_clock::now();

        for (const auto& article : articles) {
            // Time decay: recent news weighted more heavily
            auto age = now - article.timestamp;
            float decay = std::exp(-age.count() / DECAY_HALF_LIFE);

            // Source credibility weight
            float source_weight = get_source_weight(article.source);

            // Combined weight
            float weight = decay * source_weight * article.confidence;

            weighted_sum += article.sentiment * weight;
            weight_sum += weight;
        }

        return {
            .weighted_sentiment = weighted_sum / weight_sum,
            .article_count = static_cast<int>(articles.size()),
            // ... compute other fields
        };
    }

private:
    // Source credibility weights
    std::unordered_map<std::string, float> source_weights_ = {
        {"reuters", 1.0f},
        {"bloomberg", 1.0f},
        {"wsj", 0.95f},
        {"cnbc", 0.85f},
        {"benzinga", 0.80f},
        {"seekingalpha", 0.70f},
        {"motleyfool", 0.60f},
        {"unknown", 0.50f}
    };

    static constexpr float DECAY_HALF_LIFE = 6.0f * 3600.0f;  // 6 hours
};
```

### 5.3 Real-time News Processing Pipeline

```cpp
class NewsProcessor {
public:
    void start() {
        // Subscribe to news feeds
        benzinga_client_.subscribe([this](const NewsArticle& article) {
            process_article(article);
        });

        newsapi_client_.poll_interval(std::chrono::minutes(1));
        newsapi_client_.on_new_article([this](const NewsArticle& article) {
            process_article(article);
        });
    }

private:
    void process_article(const NewsArticle& article) {
        // Extract mentioned symbols
        auto symbols = extract_symbols(article.text);

        // Analyze sentiment
        auto sentiment = sentiment_analyzer_.analyze(article.headline);

        // Store in database
        for (const auto& symbol : symbols) {
            db_.insert_sentiment({
                .time = article.timestamp,
                .symbol = symbol,
                .headline = article.headline,
                .source = article.source,
                .sentiment = sentiment.score,
                .confidence = sentiment.confidence,
                .url = article.url
            });
        }

        // Publish to subscribers (dashboard, screening engine)
        zmq_publisher_.publish("news", {symbols, sentiment});
    }

    SentimentAnalyzer sentiment_analyzer_;
    BenzingaClient benzinga_client_;
    NewsAPIClient newsapi_client_;
    TimescaleDB db_;
    ZmqPublisher zmq_publisher_;
};
```

---

## 6. Signal Fusion & Decision Engine

### 6.1 Multi-Signal Fusion

```cpp
class SignalFusionEngine {
public:
    struct TradingSignal {
        std::string symbol;

        // Component signals
        float technical_signal;      // From screener: -1 to +1
        float prediction_signal;     // From TFT model: -1 to +1
        float sentiment_signal;      // From news: -1 to +1

        // Fused output
        float composite_signal;      // Weighted combination
        float confidence;            // Model confidence
        float position_size;         // Suggested position (% of portfolio)

        // Risk metrics
        float expected_return;       // Predicted % return
        float max_drawdown;          // Predicted worst case
        float sharpe_estimate;       // Risk-adjusted return

        // Trading action
        enum class Action { STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL };
        Action recommended_action;
        float stop_loss;             // Suggested stop loss price
        float take_profit;           // Suggested take profit price
    };

    TradingSignal fuse_signals(
        const std::string& symbol,
        const ScreeningFactors& screening,
        const PredictionResult& prediction,
        const AggregatedSentiment& sentiment
    ) {
        TradingSignal signal;
        signal.symbol = symbol;

        // Normalize component signals to [-1, +1]
        signal.technical_signal = normalize_technical(screening);
        signal.prediction_signal = prediction.trend_probability * 2.0f - 1.0f;
        signal.sentiment_signal = sentiment.weighted_sentiment;

        // Dynamic weight adjustment based on market regime
        auto weights = get_regime_weights(current_regime_);

        // Weighted fusion
        signal.composite_signal =
            weights.technical * signal.technical_signal +
            weights.prediction * signal.prediction_signal +
            weights.sentiment * signal.sentiment_signal;

        // Confidence from prediction model
        signal.confidence = prediction.confidence;

        // Position sizing (Kelly criterion variant)
        signal.position_size = calculate_position_size(
            signal.composite_signal,
            signal.confidence,
            prediction.lower_bound,
            prediction.upper_bound
        );

        // Risk metrics
        signal.expected_return = calculate_expected_return(prediction);
        signal.max_drawdown = estimate_max_drawdown(prediction);
        signal.sharpe_estimate = signal.expected_return / prediction_volatility;

        // Determine action
        signal.recommended_action = determine_action(signal);

        // Stop loss / take profit levels
        signal.stop_loss = calculate_stop_loss(symbol, signal);
        signal.take_profit = calculate_take_profit(symbol, signal);

        return signal;
    }

private:
    struct RegimeWeights {
        float technical;
        float prediction;
        float sentiment;
    };

    // Adjust weights based on market conditions
    RegimeWeights get_regime_weights(MarketRegime regime) {
        switch (regime) {
            case MarketRegime::TRENDING:
                return {0.30f, 0.50f, 0.20f};  // Trust predictions more
            case MarketRegime::MEAN_REVERTING:
                return {0.50f, 0.30f, 0.20f};  // Trust technicals more
            case MarketRegime::HIGH_VOLATILITY:
                return {0.25f, 0.25f, 0.50f};  // Sentiment drives moves
            case MarketRegime::LOW_VOLATILITY:
                return {0.40f, 0.40f, 0.20f};  // Balanced
            default:
                return {0.35f, 0.40f, 0.25f};  // Default balanced
        }
    }

    MarketRegime current_regime_;
};
```

### 6.2 Position Sizing (Modified Kelly Criterion)

```cpp
float calculate_position_size(
    float signal_strength,      // -1 to +1
    float confidence,           // 0 to 1
    float expected_return,      // Expected % return
    float volatility            // Expected volatility
) {
    // Kelly fraction: f* = (p * b - q) / b
    // where p = win probability, q = 1-p, b = win/loss ratio

    // Convert signal to win probability
    float win_prob = 0.5f + signal_strength * 0.25f * confidence;
    win_prob = std::clamp(win_prob, 0.3f, 0.7f);  // Cap at 30-70%

    // Estimate win/loss ratio from predictions
    float win_loss_ratio = expected_return / volatility;
    win_loss_ratio = std::clamp(win_loss_ratio, 0.5f, 3.0f);

    // Kelly fraction
    float kelly = (win_prob * win_loss_ratio - (1.0f - win_prob)) / win_loss_ratio;

    // Half-Kelly for safety (standard practice)
    float position = kelly * 0.5f;

    // Additional constraints
    position = std::clamp(position, 0.0f, 0.10f);  // Max 10% per position

    // Reduce size for low confidence
    position *= confidence;

    return position;
}
```

---

## 7. Python Dash Dashboard

### 7.1 Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARGUS TRADER DASHBOARD                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  HEADER: Portfolio Value | P&L Today | Win Rate | Active Signals │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────┐  ┌──────────────────────────────────────┐    │
│  │   TOP 20 RANKINGS    │  │         SELECTED STOCK CHART         │    │
│  │                      │  │  ┌────────────────────────────────┐  │    │
│  │  1. NVDA  ▲ 0.85    │  │  │     Candlestick + Predictions  │  │    │
│  │  2. AAPL  ▲ 0.78    │  │  │     + Confidence Intervals     │  │    │
│  │  3. TSLA  ▲ 0.72    │  │  └────────────────────────────────┘  │    │
│  │  4. MSFT  ▲ 0.68    │  │  ┌────────────────────────────────┐  │    │
│  │  5. AMZN  ▼ 0.45    │  │  │      Volume + Sentiment        │  │    │
│  │  ...                 │  │  └────────────────────────────────┘  │    │
│  │  20. META ▼ 0.32    │  │                                      │    │
│  └──────────────────────┘  └──────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────┐  ┌──────────────────────────────────────┐    │
│  │   SIGNAL DETAILS     │  │         NEWS FEED                    │    │
│  │                      │  │                                      │    │
│  │  Technical:  ████░░  │  │  • NVDA: "Record Q4 guidance..."    │    │
│  │  Prediction: █████░  │  │    Sentiment: +0.82 (Bullish)       │    │
│  │  Sentiment:  ███░░░  │  │                                      │    │
│  │  ─────────────────── │  │  • AAPL: "iPhone 16 sales beat..."  │    │
│  │  Composite:  ████░░  │  │    Sentiment: +0.65 (Bullish)       │    │
│  │                      │  │                                      │    │
│  │  Action: STRONG BUY  │  │  • TSLA: "Robotaxi delay..."        │    │
│  │  Position: 5.2%      │  │    Sentiment: -0.45 (Bearish)       │    │
│  │  Stop: $142.50       │  │                                      │    │
│  │  Target: $168.00     │  │                                      │    │
│  └──────────────────────┘  └──────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ALERTS: ⚠️ NVDA approaching take-profit | 🔔 New signal: AMD    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Dashboard Implementation

```python
# dashboard/app.py

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zmq
import json
import pandas as pd
from datetime import datetime, timedelta

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    update_title=None  # Prevent "Updating..." in title
)

# ZeroMQ subscriber for real-time data from C++ engine
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all

# Layout
app.layout = dbc.Container([
    # Header row
    dbc.Row([
        dbc.Col([
            html.H1("ArgusTrader", className="text-primary"),
            html.Span(id="connection-status", className="badge bg-success")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="portfolio-value", className="text-success"),
                    html.P("Portfolio Value", className="text-muted")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="daily-pnl", className="text-success"),
                    html.P("Today's P&L", className="text-muted")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="win-rate", className="text-info"),
                    html.P("Win Rate", className="text-muted")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="active-signals", className="text-warning"),
                    html.P("Active Signals", className="text-muted")
                ])
            ])
        ], width=2),
    ], className="mb-4"),

    # Main content row
    dbc.Row([
        # Left column: Rankings
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top 20 Rankings"),
                dbc.CardBody([
                    html.Div(id="rankings-table")
                ])
            ])
        ], width=3),

        # Center column: Charts
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span(id="selected-symbol", className="h5"),
                    dbc.ButtonGroup([
                        dbc.Button("1D", id="btn-1d", size="sm"),
                        dbc.Button("1W", id="btn-1w", size="sm"),
                        dbc.Button("1M", id="btn-1m", size="sm"),
                    ], className="float-end")
                ]),
                dbc.CardBody([
                    dcc.Graph(id="price-chart", config={'displayModeBar': False}),
                    dcc.Graph(id="volume-sentiment-chart", config={'displayModeBar': False})
                ])
            ])
        ], width=6),

        # Right column: Signal details + News
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Signal Analysis"),
                dbc.CardBody([
                    html.Div(id="signal-details")
                ])
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("News Feed"),
                dbc.CardBody([
                    html.Div(id="news-feed", style={"maxHeight": "300px", "overflowY": "auto"})
                ])
            ])
        ], width=3)
    ]),

    # Alerts row
    dbc.Row([
        dbc.Col([
            dbc.Alert(id="alerts-bar", color="info", className="mt-3")
        ])
    ]),

    # Interval component for updates
    dcc.Interval(id="interval-fast", interval=1000),   # 1 second for prices
    dcc.Interval(id="interval-slow", interval=5000),   # 5 seconds for rankings

    # Store for selected symbol
    dcc.Store(id="store-selected-symbol", data="NVDA")

], fluid=True)


@app.callback(
    [Output("rankings-table", "children"),
     Output("portfolio-value", "children"),
     Output("daily-pnl", "children"),
     Output("active-signals", "children")],
    Input("interval-slow", "n_intervals")
)
def update_rankings(_):
    """Fetch latest rankings from C++ engine."""
    try:
        # Non-blocking receive
        message = socket.recv_string(flags=zmq.NOBLOCK)
        data = json.loads(message)

        if data["type"] == "rankings":
            rankings = data["rankings"]

            # Build table
            rows = []
            for i, stock in enumerate(rankings[:20]):
                signal_color = "success" if stock["composite_score"] > 0.5 else \
                              "warning" if stock["composite_score"] > 0 else "danger"
                arrow = "▲" if stock["composite_score"] > 0.5 else "▼"

                rows.append(
                    html.Tr([
                        html.Td(f"{i+1}"),
                        html.Td(stock["symbol"], className="fw-bold"),
                        html.Td(f"{arrow} {stock['composite_score']:.2f}",
                               className=f"text-{signal_color}")
                    ], id={"type": "ranking-row", "symbol": stock["symbol"]},
                       style={"cursor": "pointer"})
                )

            table = dbc.Table([
                html.Thead(html.Tr([
                    html.Th("#"), html.Th("Symbol"), html.Th("Signal")
                ])),
                html.Tbody(rows)
            ], striped=True, hover=True, size="sm")

            return (
                table,
                f"${data.get('portfolio_value', 0):,.2f}",
                f"${data.get('daily_pnl', 0):+,.2f}",
                str(data.get('active_signals', 0))
            )

    except zmq.Again:
        pass  # No message available

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output("price-chart", "figure"),
    [Input("interval-fast", "n_intervals"),
     Input("store-selected-symbol", "data")]
)
def update_price_chart(_, symbol):
    """Update candlestick chart with predictions."""
    # Fetch data from C++ engine
    data = fetch_stock_data(symbol)

    fig = make_subplots(rows=1, cols=1)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data["timestamps"],
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        name="Price"
    ))

    # Prediction line
    if "predictions" in data:
        fig.add_trace(go.Scatter(
            x=data["pred_timestamps"],
            y=data["predictions"],
            mode="lines",
            name="Prediction",
            line=dict(color="cyan", dash="dash")
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=data["pred_timestamps"] + data["pred_timestamps"][::-1],
            y=data["upper_bound"] + data["lower_bound"][::-1],
            fill="toself",
            fillcolor="rgba(0, 255, 255, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% CI"
        ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=350,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig


@app.callback(
    Output("signal-details", "children"),
    Input("store-selected-symbol", "data")
)
def update_signal_details(symbol):
    """Display signal breakdown for selected stock."""
    signal = fetch_signal(symbol)

    if not signal:
        return html.P("No signal available")

    action_colors = {
        "STRONG_BUY": "success",
        "BUY": "info",
        "HOLD": "secondary",
        "SELL": "warning",
        "STRONG_SELL": "danger"
    }

    return html.Div([
        # Signal bars
        html.Div([
            html.Label("Technical"),
            dbc.Progress(value=abs(signal["technical"]) * 100,
                        color="success" if signal["technical"] > 0 else "danger",
                        className="mb-2")
        ]),
        html.Div([
            html.Label("Prediction"),
            dbc.Progress(value=abs(signal["prediction"]) * 100,
                        color="success" if signal["prediction"] > 0 else "danger",
                        className="mb-2")
        ]),
        html.Div([
            html.Label("Sentiment"),
            dbc.Progress(value=abs(signal["sentiment"]) * 100,
                        color="success" if signal["sentiment"] > 0 else "danger",
                        className="mb-2")
        ]),
        html.Hr(),
        html.Div([
            html.Label("Composite"),
            dbc.Progress(value=abs(signal["composite"]) * 100,
                        color="success" if signal["composite"] > 0 else "danger",
                        className="mb-3")
        ]),

        # Action badge
        dbc.Badge(signal["action"], color=action_colors.get(signal["action"], "secondary"),
                 className="me-2 mb-3", style={"fontSize": "1.2em"}),

        # Position details
        html.Table([
            html.Tr([html.Td("Position Size:"), html.Td(f"{signal['position_size']:.1%}")]),
            html.Tr([html.Td("Stop Loss:"), html.Td(f"${signal['stop_loss']:.2f}")]),
            html.Tr([html.Td("Take Profit:"), html.Td(f"${signal['take_profit']:.2f}")]),
            html.Tr([html.Td("Expected Return:"), html.Td(f"{signal['expected_return']:+.2%}")]),
            html.Tr([html.Td("Confidence:"), html.Td(f"{signal['confidence']:.1%}")]),
        ], className="table table-sm")
    ])


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8050)
```

### 7.3 C++ ↔ Python Communication (ZeroMQ)

```cpp
// engine/zmq_publisher.hpp

#include <zmq.hpp>
#include <nlohmann/json.hpp>

class DashboardPublisher {
public:
    DashboardPublisher(const std::string& endpoint = "tcp://*:5555")
        : context_(1), socket_(context_, zmq::socket_type::pub)
    {
        socket_.bind(endpoint);
    }

    void publish_rankings(
        const std::array<ScoredStock, 20>& rankings,
        float portfolio_value,
        float daily_pnl,
        int active_signals
    ) {
        nlohmann::json j;
        j["type"] = "rankings";
        j["timestamp"] = get_timestamp_ms();
        j["portfolio_value"] = portfolio_value;
        j["daily_pnl"] = daily_pnl;
        j["active_signals"] = active_signals;

        j["rankings"] = nlohmann::json::array();
        for (const auto& stock : rankings) {
            j["rankings"].push_back({
                {"symbol", stock.symbol},
                {"composite_score", stock.composite_score},
                {"confidence", stock.confidence},
                {"factors", {
                    {"momentum", stock.factors.price_momentum_5d},
                    {"volume", stock.factors.volume_surge},
                    {"sentiment", stock.factors.news_sentiment_24h}
                }}
            });
        }

        socket_.send(zmq::buffer(j.dump()), zmq::send_flags::none);
    }

    void publish_signal(const TradingSignal& signal) {
        nlohmann::json j;
        j["type"] = "signal";
        j["symbol"] = signal.symbol;
        j["technical"] = signal.technical_signal;
        j["prediction"] = signal.prediction_signal;
        j["sentiment"] = signal.sentiment_signal;
        j["composite"] = signal.composite_signal;
        j["confidence"] = signal.confidence;
        j["position_size"] = signal.position_size;
        j["action"] = action_to_string(signal.recommended_action);
        j["stop_loss"] = signal.stop_loss;
        j["take_profit"] = signal.take_profit;
        j["expected_return"] = signal.expected_return;

        socket_.send(zmq::buffer(j.dump()), zmq::send_flags::none);
    }

    void publish_alert(const std::string& message, const std::string& severity) {
        nlohmann::json j;
        j["type"] = "alert";
        j["message"] = message;
        j["severity"] = severity;
        j["timestamp"] = get_timestamp_ms();

        socket_.send(zmq::buffer(j.dump()), zmq::send_flags::none);
    }

private:
    zmq::context_t context_;
    zmq::socket_t socket_;
};
```

---

## 8. Infrastructure & Deployment

### 8.1 Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | 8 cores | 16+ cores | Intel Xeon or AMD EPYC |
| **RAM** | 32 GB | 64 GB | For market data buffers |
| **GPU** | RTX 3080 | RTX 4090 / A100 | TensorRT inference |
| **Storage** | 500GB NVMe | 2TB NVMe | TimescaleDB + tick data |
| **Network** | 1 Gbps | 10 Gbps | Low-latency connection |

### 8.2 Software Dependencies

```cmake
# CMakeLists.txt

cmake_minimum_required(VERSION 3.20)
project(ArgusTrader LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

# Dependencies
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(cppzmq REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)
find_package(libpqxx REQUIRED)  # PostgreSQL client
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system asio)

# Main executable
add_executable(argus_engine
    src/main.cpp
    src/market_data/polygon_client.cpp
    src/market_data/alpaca_client.cpp
    src/screener/stock_screener.cpp
    src/predictor/tensorrt_predictor.cpp
    src/sentiment/sentiment_analyzer.cpp
    src/fusion/signal_fusion.cpp
    src/dashboard/zmq_publisher.cpp
    src/database/timescale_client.cpp
)

target_link_libraries(argus_engine
    CUDA::cudart
    nvinfer
    nvonnxparser
    cppzmq
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    pqxx
    OpenSSL::SSL
    CURL::libcurl
    Boost::system
    Boost::asio
)

# Compiler optimizations
target_compile_options(argus_engine PRIVATE
    -O3
    -march=native
    -ffast-math
    -funroll-loops
)
```

### 8.3 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION DEPLOYMENT                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    COMPUTE SERVER (GPU)                           │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │   │
│  │  │ ArgusEngine    │  │ Python Dash    │  │  TimescaleDB   │     │   │
│  │  │ (C++ Core)     │  │  Dashboard     │  │   Database     │     │   │
│  │  │                │  │                │  │                │     │   │
│  │  │ • Data Ingest  │◄─┤ • Web UI       │  │ • Tick Data    │     │   │
│  │  │ • Screener     │  │ • Callbacks    │  │ • Sentiment    │     │   │
│  │  │ • TensorRT     │─►│ • Charts       │  │ • Signals      │     │   │
│  │  │ • Sentiment    │  │                │  │                │     │   │
│  │  │ • Signals      │  │ Port: 8050     │  │ Port: 5432     │     │   │
│  │  │                │  │                │  │                │     │   │
│  │  │ ZMQ: 5555      │  └────────────────┘  └────────────────┘     │   │
│  │  └────────────────┘                                              │   │
│  │           │                                                       │   │
│  │           │ RTX 4090 GPU                                         │   │
│  │           │ • TFT Inference                                      │   │
│  │           │ • FinBERT Sentiment                                  │   │
│  │           │ • Batch Processing                                   │   │
│  └───────────┼──────────────────────────────────────────────────────┘   │
│              │                                                           │
│              │ Internet                                                  │
│              │                                                           │
│  ┌───────────┼──────────────────────────────────────────────────────┐   │
│  │           │           EXTERNAL SERVICES                           │   │
│  │  ┌────────┴───────┐  ┌────────────────┐  ┌────────────────┐     │   │
│  │  │   Polygon.io   │  │   NewsAPI      │  │    Alpaca      │     │   │
│  │  │  Market Data   │  │  News Feed     │  │   Brokerage    │     │   │
│  │  │   WebSocket    │  │   REST API     │  │   Trading      │     │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.4 Docker Compose Setup

```yaml
# docker-compose.yml

version: '3.8'

services:
  argus-engine:
    build:
      context: ./engine
      dockerfile: Dockerfile.gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - POLYGON_API_KEY=${POLYGON_API_KEY}
      - NEWSAPI_KEY=${NEWSAPI_KEY}
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
    volumes:
      - ./models:/app/models:ro
      - ./config:/app/config:ro
    ports:
      - "5555:5555"  # ZeroMQ
    depends_on:
      - timescaledb
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  dashboard:
    build:
      context: ./dashboard
    environment:
      - ZMQ_ENDPOINT=tcp://argus-engine:5555
      - TIMESCALE_HOST=timescaledb
    ports:
      - "8050:8050"
    depends_on:
      - argus-engine

  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      - POSTGRES_USER=argus
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=argus_trader
    volumes:
      - timescale_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  timescale_data:
```

---

## 9. Risk Management

### 9.1 Position Limits

```cpp
struct RiskLimits {
    // Per-position limits
    float max_position_pct = 0.10f;         // 10% max per stock
    float max_position_value = 50000.0f;    // $50K absolute max

    // Portfolio limits
    float max_sector_exposure = 0.30f;      // 30% max per sector
    float max_total_exposure = 0.80f;       // 80% max invested (20% cash)
    int max_open_positions = 20;

    // Loss limits
    float daily_loss_limit_pct = 0.02f;     // 2% max daily loss
    float position_stop_loss_pct = 0.05f;   // 5% stop loss per position
    float trailing_stop_pct = 0.03f;        // 3% trailing stop

    // Volatility adjustments
    float high_vix_reduction = 0.50f;       // Reduce positions 50% when VIX > 30
    float vix_threshold = 30.0f;
};

class RiskManager {
public:
    bool validate_order(const Order& order, const Portfolio& portfolio) {
        // Check position size
        if (order.value / portfolio.total_value > limits_.max_position_pct) {
            log_rejection("Position size exceeds limit");
            return false;
        }

        // Check sector exposure
        float sector_exposure = calculate_sector_exposure(
            order.symbol, portfolio) + order.value;
        if (sector_exposure / portfolio.total_value > limits_.max_sector_exposure) {
            log_rejection("Sector exposure exceeds limit");
            return false;
        }

        // Check daily loss
        if (portfolio.daily_pnl / portfolio.total_value < -limits_.daily_loss_limit_pct) {
            log_rejection("Daily loss limit reached");
            return false;
        }

        // Check VIX adjustment
        if (current_vix_ > limits_.vix_threshold) {
            // Reduce all position sizes
            order.quantity *= (1.0f - limits_.high_vix_reduction);
        }

        return true;
    }

private:
    RiskLimits limits_;
    float current_vix_;
};
```

### 9.2 Circuit Breakers

```cpp
class CircuitBreaker {
public:
    enum class State { CLOSED, OPEN, HALF_OPEN };

    bool allow_trade() {
        auto now = std::chrono::steady_clock::now();

        switch (state_) {
            case State::CLOSED:
                return true;

            case State::OPEN:
                if (now - last_failure_ > cooldown_period_) {
                    state_ = State::HALF_OPEN;
                    return true;
                }
                return false;

            case State::HALF_OPEN:
                return true;  // Allow one trade to test
        }
        return false;
    }

    void record_success() {
        if (state_ == State::HALF_OPEN) {
            state_ = State::CLOSED;
            failure_count_ = 0;
        }
    }

    void record_failure() {
        failure_count_++;
        last_failure_ = std::chrono::steady_clock::now();

        if (failure_count_ >= failure_threshold_) {
            state_ = State::OPEN;
            log_alert("Circuit breaker OPEN - trading halted");
        }
    }

private:
    State state_ = State::CLOSED;
    int failure_count_ = 0;
    int failure_threshold_ = 5;
    std::chrono::seconds cooldown_period_{300};  // 5 minutes
    std::chrono::steady_clock::time_point last_failure_;
};
```

### 9.3 Audit Logging

```cpp
struct AuditLog {
    uint64_t timestamp_ns;
    std::string event_type;      // "SIGNAL", "ORDER", "FILL", "RISK_CHECK"
    std::string symbol;
    nlohmann::json details;
    std::string user;
    std::string source_ip;
};

class AuditLogger {
public:
    void log_signal(const TradingSignal& signal) {
        AuditLog entry{
            .timestamp_ns = get_nanos(),
            .event_type = "SIGNAL",
            .symbol = signal.symbol,
            .details = {
                {"composite", signal.composite_signal},
                {"action", action_to_string(signal.recommended_action)},
                {"confidence", signal.confidence}
            }
        };
        write_to_log(entry);
    }

    void log_order(const Order& order, const std::string& decision) {
        AuditLog entry{
            .timestamp_ns = get_nanos(),
            .event_type = "ORDER",
            .symbol = order.symbol,
            .details = {
                {"side", order.side},
                {"quantity", order.quantity},
                {"price", order.price},
                {"decision", decision}  // "APPROVED", "REJECTED", reason
            }
        };
        write_to_log(entry);
    }

private:
    void write_to_log(const AuditLog& entry) {
        // Write to TimescaleDB for queryability
        db_.insert("audit_log", entry);

        // Write to append-only file for compliance
        file_ << entry.to_json() << "\n";
        file_.flush();
    }

    TimescaleDB db_;
    std::ofstream file_;
};
```

---

## 10. Development Roadmap

### Phase 1: Foundation (Weeks 1-4)
- [ ] Set up C++ build system with CMake
- [ ] Implement Polygon.io WebSocket client
- [ ] Create basic data structures and ring buffers
- [ ] Set up TimescaleDB schema
- [ ] Basic ZeroMQ publisher

### Phase 2: Screening Engine (Weeks 5-8)
- [ ] Implement technical indicator calculations
- [ ] Build multi-factor screener
- [ ] Create universe filtering logic
- [ ] Add real-time ranking updates
- [ ] Unit tests for screener

### Phase 3: Prediction Engine (Weeks 9-12)
- [ ] Train TFT model on historical data (Python)
- [ ] Export to ONNX format
- [ ] Convert to TensorRT engine
- [ ] Integrate TensorRT inference in C++
- [ ] Benchmark latency

### Phase 4: Sentiment Analysis (Weeks 13-16)
- [ ] Integrate NewsAPI and Benzinga
- [ ] Deploy FinBERT model (TensorRT)
- [ ] Build news aggregation pipeline
- [ ] Implement sentiment scoring
- [ ] Test sentiment → signal correlation

### Phase 5: Signal Fusion (Weeks 17-18)
- [ ] Implement signal fusion engine
- [ ] Add position sizing logic
- [ ] Build risk manager
- [ ] Create audit logging

### Phase 6: Dashboard (Weeks 19-22)
- [ ] Build Python Dash interface
- [ ] Implement real-time charts
- [ ] Add signal visualization
- [ ] Create alerts system
- [ ] User testing and refinement

### Phase 7: Paper Trading (Weeks 23-26)
- [ ] Integrate Alpaca paper trading
- [ ] Run simulated trading
- [ ] Collect performance metrics
- [ ] Tune parameters based on results
- [ ] Stress testing

### Phase 8: Production (Weeks 27-30)
- [ ] Deploy to production server
- [ ] Set up monitoring and alerting
- [ ] Implement gradual rollout
- [ ] Documentation and runbooks
- [ ] Go-live with real capital (small allocation)

---

## 11. Appendices

### A. API Keys Required

| Service | Purpose | Estimated Cost |
|---------|---------|----------------|
| Polygon.io | Real-time market data | $199/mo (Starter) |
| Alpaca | Trading + data | Free (paper), $0 (live) |
| NewsAPI | News aggregation | $449/mo (Business) |
| Benzinga | Premium news wire | Custom pricing |
| Alpha Vantage | Backup data | $49/mo (Premium) |

### B. Regulatory Considerations

- **Pattern Day Trader Rule**: Minimum $25,000 account balance for frequent trading
- **SEC Regulations**: Algorithmic trading disclosure requirements
- **FINRA Rules**: Best execution requirements
- **Data Licensing**: Ensure proper licensing for market data redistribution

### C. Backtesting Framework

```python
# backtesting/backtest_engine.py

class BacktestEngine:
    def __init__(self, start_date, end_date, initial_capital=100000):
        self.start_date = start_date
        self.end_date = end_date
        self.capital = initial_capital

    def run(self, strategy):
        results = []
        for date in self.trading_days():
            # Get signals from strategy
            signals = strategy.generate_signals(date)

            # Execute trades
            fills = self.execute_orders(signals)

            # Update portfolio
            self.portfolio.update(fills)

            # Record metrics
            results.append({
                'date': date,
                'portfolio_value': self.portfolio.value,
                'cash': self.portfolio.cash,
                'positions': len(self.portfolio.positions),
                'daily_return': self.portfolio.daily_return
            })

        return BacktestResults(results)

    def calculate_metrics(self, results):
        return {
            'total_return': results.total_return,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'avg_trade_duration': results.avg_trade_duration
        }
```

### D. Performance Benchmarks

| Operation | Target | Achieved | Notes |
|-----------|--------|----------|-------|
| Tick ingestion | < 100 μs | TBD | Per message |
| Technical indicators | < 500 μs | TBD | Per stock |
| TFT inference (batch=20) | < 500 μs | TBD | TensorRT FP16 |
| Sentiment analysis | < 5 ms | TBD | FinBERT |
| Signal fusion | < 100 μs | TBD | Per stock |
| End-to-end | < 10 ms | TBD | Data → Signal |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Nov 2025 | Craig Giannelli, Claude | Initial design document |

---

**Built by Craig Giannelli and Claude Code**
