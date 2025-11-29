"""
ArgusTrader Dashboard Configuration
"""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Dashboard Settings
    dash_host: str = "127.0.0.1"
    dash_port: int = 8050
    dash_debug: bool = True

    # ZeroMQ Connection to C++ Engine
    zmq_engine_host: str = "127.0.0.1"
    zmq_engine_port: int = 5555
    zmq_timeout_ms: int = 1000

    # Polygon API (for direct REST calls if needed)
    polygon_api_key: str = ""

    # Database (optional - for when TimescaleDB is ready)
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "argus"
    db_user: str = "argus"
    db_password: str = ""

    # Redis (optional - for caching)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # UI Settings
    update_interval_ms: int = 1000  # Chart update interval
    max_chart_points: int = 500  # Max data points on charts

    @property
    def zmq_engine_address(self) -> str:
        return f"tcp://{self.zmq_engine_host}:{self.zmq_engine_port}"

    @property
    def db_url(self) -> str:
        if not self.db_password:
            return ""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            dash_host=os.getenv("DASH_HOST", "127.0.0.1"),
            dash_port=int(os.getenv("DASH_PORT", "8050")),
            dash_debug=os.getenv("DASH_DEBUG", "true").lower() == "true",
            zmq_engine_host=os.getenv("ZMQ_ENGINE_HOST", "127.0.0.1"),
            zmq_engine_port=int(os.getenv("ZMQ_ENGINE_PORT", "5555")),
            zmq_timeout_ms=int(os.getenv("ZMQ_TIMEOUT_MS", "1000")),
            polygon_api_key=os.getenv("POLYGON_API_KEY", ""),
            db_host=os.getenv("DB_HOST", "localhost"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            db_name=os.getenv("DB_NAME", "argus"),
            db_user=os.getenv("DB_USER", "argus"),
            db_password=os.getenv("DB_PASSWORD", ""),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            update_interval_ms=int(os.getenv("UPDATE_INTERVAL_MS", "1000")),
            max_chart_points=int(os.getenv("MAX_CHART_POINTS", "500")),
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()
