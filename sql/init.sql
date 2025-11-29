-- ============================================================================
-- ArgusTrader Database Initialization
-- ============================================================================
-- This script runs first to set up extensions and basic configuration
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create schema if needed
CREATE SCHEMA IF NOT EXISTS argus;

-- Set default search path
ALTER DATABASE argus_trader SET search_path TO public, argus;

-- Performance settings (these may be overridden in docker-compose)
ALTER SYSTEM SET shared_preload_libraries = 'timescaledb';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';

-- TimescaleDB specific settings
ALTER SYSTEM SET timescaledb.max_background_workers = 8;

-- Logging settings for debugging (can be disabled in production)
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1s

-- Notify that init is complete
DO $$
BEGIN
    RAISE NOTICE 'ArgusTrader database initialization complete';
END $$;
