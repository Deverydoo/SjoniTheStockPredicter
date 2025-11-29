-- ============================================================================
-- ArgusTrader Database Schema
-- ============================================================================
-- TimescaleDB schema for storing market data, signals, and audit logs
-- ============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================================
-- Market Data Tables
-- ============================================================================

-- Raw tick data
CREATE TABLE IF NOT EXISTS market_ticks (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    price       DOUBLE PRECISION NOT NULL,
    volume      BIGINT NOT NULL,
    exchange    CHAR(1),
    conditions  INTEGER[],

    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('market_ticks', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create index for symbol lookups
CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time
    ON market_ticks (symbol, time DESC);

-- ============================================================================
-- OHLCV Aggregated Bars
-- ============================================================================

-- 1-minute bars
CREATE TABLE IF NOT EXISTS ohlcv_1m (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT NOT NULL,
    vwap        DOUBLE PRECISION,
    trade_count INTEGER,

    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('ohlcv_1m', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_symbol_time
    ON ohlcv_1m (symbol, time DESC);

-- 5-minute bars
CREATE TABLE IF NOT EXISTS ohlcv_5m (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT NOT NULL,
    vwap        DOUBLE PRECISION,
    trade_count INTEGER,

    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('ohlcv_5m', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_5m_symbol_time
    ON ohlcv_5m (symbol, time DESC);

-- Daily bars
CREATE TABLE IF NOT EXISTS ohlcv_1d (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT NOT NULL,
    vwap        DOUBLE PRECISION,
    trade_count INTEGER,
    adj_close   DOUBLE PRECISION,

    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('ohlcv_1d', 'time',
    chunk_time_interval => INTERVAL '1 year',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_1d_symbol_time
    ON ohlcv_1d (symbol, time DESC);

-- ============================================================================
-- News & Sentiment Tables
-- ============================================================================

-- News articles with sentiment
CREATE TABLE IF NOT EXISTS news_sentiment (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    headline    TEXT NOT NULL,
    source      TEXT NOT NULL,
    sentiment   DOUBLE PRECISION NOT NULL,  -- -1.0 to 1.0
    confidence  DOUBLE PRECISION NOT NULL,  -- 0.0 to 1.0
    url         TEXT,
    article_id  TEXT UNIQUE,

    PRIMARY KEY (time, symbol, article_id)
);

SELECT create_hypertable('news_sentiment', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_news_symbol_time
    ON news_sentiment (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_news_source
    ON news_sentiment (source, time DESC);

-- Aggregated sentiment scores
CREATE TABLE IF NOT EXISTS sentiment_aggregated (
    time                TIMESTAMPTZ NOT NULL,
    symbol              TEXT NOT NULL,
    weighted_sentiment  DOUBLE PRECISION NOT NULL,
    sentiment_momentum  DOUBLE PRECISION,
    article_count       INTEGER NOT NULL,
    attention_score     DOUBLE PRECISION,

    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('sentiment_aggregated', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- Stock Screening Tables
-- ============================================================================

-- Screener rankings
CREATE TABLE IF NOT EXISTS screener_rankings (
    time                TIMESTAMPTZ NOT NULL,
    symbol              TEXT NOT NULL,
    rank                INTEGER NOT NULL,
    composite_score     DOUBLE PRECISION NOT NULL,
    momentum_score      DOUBLE PRECISION,
    volume_score        DOUBLE PRECISION,
    volatility_score    DOUBLE PRECISION,
    sentiment_score     DOUBLE PRECISION,
    confidence          DOUBLE PRECISION,

    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('screener_rankings', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_rankings_time_rank
    ON screener_rankings (time DESC, rank);

-- ============================================================================
-- Trading Signals Tables
-- ============================================================================

-- Generated trading signals
CREATE TABLE IF NOT EXISTS trading_signals (
    time                TIMESTAMPTZ NOT NULL,
    symbol              TEXT NOT NULL,

    -- Component signals
    technical_signal    DOUBLE PRECISION NOT NULL,
    prediction_signal   DOUBLE PRECISION NOT NULL,
    sentiment_signal    DOUBLE PRECISION NOT NULL,

    -- Fused output
    composite_signal    DOUBLE PRECISION NOT NULL,
    confidence          DOUBLE PRECISION NOT NULL,
    position_size       DOUBLE PRECISION NOT NULL,

    -- Risk metrics
    expected_return     DOUBLE PRECISION,
    max_drawdown        DOUBLE PRECISION,
    sharpe_estimate     DOUBLE PRECISION,

    -- Action
    action              TEXT NOT NULL,  -- STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    stop_loss           DOUBLE PRECISION,
    take_profit         DOUBLE PRECISION,

    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('trading_signals', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_time
    ON trading_signals (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_signals_action
    ON trading_signals (action, time DESC);

-- ============================================================================
-- Predictions Tables
-- ============================================================================

-- Model predictions
CREATE TABLE IF NOT EXISTS predictions (
    time                TIMESTAMPTZ NOT NULL,
    symbol              TEXT NOT NULL,
    horizon             INTEGER NOT NULL,  -- Steps ahead
    predicted_price     DOUBLE PRECISION NOT NULL,
    lower_bound         DOUBLE PRECISION,  -- 10th percentile
    upper_bound         DOUBLE PRECISION,  -- 90th percentile
    trend_probability   DOUBLE PRECISION,  -- P(up)
    confidence          DOUBLE PRECISION NOT NULL,
    model_version       TEXT,

    PRIMARY KEY (time, symbol, horizon)
);

SELECT create_hypertable('predictions', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- Portfolio & Orders Tables
-- ============================================================================

-- Portfolio positions
CREATE TABLE IF NOT EXISTS positions (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    avg_cost        DOUBLE PRECISION NOT NULL,
    current_price   DOUBLE PRECISION,
    unrealized_pnl  DOUBLE PRECISION,
    opened_at       TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol
    ON positions (symbol);

-- Order history
CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,  -- BUY, SELL
    quantity        INTEGER NOT NULL,
    order_type      TEXT NOT NULL,  -- MARKET, LIMIT, STOP
    limit_price     DOUBLE PRECISION,
    stop_price      DOUBLE PRECISION,
    status          TEXT NOT NULL,  -- PENDING, FILLED, CANCELLED, REJECTED
    filled_qty      INTEGER DEFAULT 0,
    filled_avg_price DOUBLE PRECISION,
    signal_id       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    filled_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol_time
    ON orders (symbol, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_orders_status
    ON orders (status, created_at DESC);

-- ============================================================================
-- Audit & Logging Tables
-- ============================================================================

-- Audit log
CREATE TABLE IF NOT EXISTS audit_log (
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type      TEXT NOT NULL,
    symbol          TEXT,
    details         JSONB NOT NULL,
    user_id         TEXT,
    source_ip       TEXT,

    PRIMARY KEY (time, event_type)
);

SELECT create_hypertable('audit_log', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_audit_event_type
    ON audit_log (event_type, time DESC);

CREATE INDEX IF NOT EXISTS idx_audit_symbol
    ON audit_log (symbol, time DESC) WHERE symbol IS NOT NULL;

-- ============================================================================
-- Reference Data Tables
-- ============================================================================

-- Stock universe
CREATE TABLE IF NOT EXISTS stock_universe (
    symbol          TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    sector          TEXT,
    industry        TEXT,
    market_cap      BIGINT,
    exchange        TEXT,
    is_active       BOOLEAN DEFAULT TRUE,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_universe_sector
    ON stock_universe (sector) WHERE is_active = TRUE;

-- ============================================================================
-- Continuous Aggregates (Materialized Views)
-- ============================================================================

-- Hourly OHLCV from 1-minute bars
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS time,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    sum(volume * vwap) / NULLIF(sum(volume), 0) AS vwap,
    sum(trade_count) AS trade_count
FROM ohlcv_1m
GROUP BY time_bucket('1 hour', time), symbol
WITH NO DATA;

-- Refresh policy for hourly aggregates
SELECT add_continuous_aggregate_policy('ohlcv_1h',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ============================================================================
-- Data Retention Policies
-- ============================================================================

-- Keep tick data for 7 days
SELECT add_retention_policy('market_ticks', INTERVAL '7 days', if_not_exists => TRUE);

-- Keep 1-minute bars for 30 days
SELECT add_retention_policy('ohlcv_1m', INTERVAL '30 days', if_not_exists => TRUE);

-- Keep 5-minute bars for 90 days
SELECT add_retention_policy('ohlcv_5m', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep news sentiment for 90 days
SELECT add_retention_policy('news_sentiment', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep audit logs for 1 year
SELECT add_retention_policy('audit_log', INTERVAL '1 year', if_not_exists => TRUE);

-- ============================================================================
-- Compression Policies
-- ============================================================================

-- Enable compression on older data
ALTER TABLE ohlcv_1m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('ohlcv_1m', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE ohlcv_5m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('ohlcv_5m', INTERVAL '30 days', if_not_exists => TRUE);

-- ============================================================================
-- Useful Functions
-- ============================================================================

-- Function to get latest price for a symbol
CREATE OR REPLACE FUNCTION get_latest_price(p_symbol TEXT)
RETURNS DOUBLE PRECISION AS $$
    SELECT close
    FROM ohlcv_1m
    WHERE symbol = p_symbol
    ORDER BY time DESC
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- Function to get latest signal for a symbol
CREATE OR REPLACE FUNCTION get_latest_signal(p_symbol TEXT)
RETURNS TABLE (
    time TIMESTAMPTZ,
    composite_signal DOUBLE PRECISION,
    action TEXT,
    confidence DOUBLE PRECISION
) AS $$
    SELECT time, composite_signal, action, confidence
    FROM trading_signals
    WHERE symbol = p_symbol
    ORDER BY time DESC
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- ============================================================================
-- Grants
-- ============================================================================

-- Grant permissions to argus user (adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO argus;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO argus;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO argus;
