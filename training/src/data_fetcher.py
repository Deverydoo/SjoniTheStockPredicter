"""
Historical market data fetchers for training ML models.

Supports:
- Polygon.io (2 years free, more with paid)
- Yahoo Finance (20+ years, free)
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Yahoo Finance Fetcher (FREE - 20+ years of data)
# =============================================================================

class YahooDataFetcher:
    """
    Fetch historical OHLCV data from Yahoo Finance via yfinance.

    Advantages:
    - FREE with no API key required
    - 20+ years of historical data
    - Adjusted prices for splits/dividends
    - No rate limiting issues

    Limitations:
    - Data quality not as high as paid sources
    - Occasional gaps or errors
    - No real-time data
    """

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

    def get_daily_bars(
        self,
        symbol: str,
        years: int = 5,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for a symbol.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            years: Years of history to fetch (ignored if start_date provided)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching {symbol} from {start_date} to {end_date}")

        ticker = self.yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Reset index to get date as column
        df = df.reset_index()

        # Rename columns to match our format
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        df["symbol"] = symbol

        # Select and order columns
        cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Remove timezone info if present
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        logger.info(f"Got {len(df)} bars for {symbol}")
        return df

    def get_multiple_symbols(
        self,
        symbols: list[str],
        years: int = 5,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                df = self.get_daily_bars(symbol, years=years)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        return results


def download_yahoo_data(
    symbols: list[str],
    output_dir: str = "data",
    years: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Download historical data from Yahoo Finance for multiple symbols.

    This is the RECOMMENDED method for getting free historical data.

    Args:
        symbols: List of stock tickers
        output_dir: Directory to save parquet files
        years: Years of history to fetch (max ~20)

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fetcher = YahooDataFetcher()
    results = {}

    for symbol in symbols:
        logger.info(f"Fetching {symbol} from Yahoo Finance...")

        try:
            df = fetcher.get_daily_bars(symbol, years=years)

            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue

            # Save to parquet
            filename = output_path / f"{symbol}_day.parquet"
            df.to_parquet(filename, index=False)
            logger.info(f"Saved {len(df)} rows to {filename}")

            results[symbol] = df

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            continue

    return results


# =============================================================================
# Polygon.io Fetcher (Paid for more than 2 years)
# =============================================================================


class PolygonDataFetcher:
    """
    Fetch historical OHLCV data from Polygon.io REST API.

    Free tier limits:
    - 5 API calls per minute
    - 2 years of historical data
    - End-of-day data only (no intraday on free)

    Paid tiers get:
    - Higher rate limits
    - Intraday bars (1min, 5min, etc.)
    - Real-time data
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")

        self.client = httpx.Client(timeout=30.0)
        self._last_request_time = 0
        self._min_request_interval = 12.5  # 5 requests per minute = 12 seconds between

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make authenticated GET request."""
        self._rate_limit()

        params = params or {}
        params["apiKey"] = self.api_key

        url = f"{self.BASE_URL}{endpoint}"
        logger.debug(f"GET {url}")

        response = self.client.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def get_aggregates(
        self,
        symbol: str,
        multiplier: int = 1,
        timespan: str = "day",  # minute, hour, day, week, month
        from_date: str = None,
        to_date: str = None,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """
        Fetch aggregate bars (OHLCV) for a symbol.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            multiplier: Size of the timespan multiplier (e.g., 1 for 1-day, 5 for 5-minute)
            timespan: minute, hour, day, week, month, quarter, year
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Max results per request

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, vwap, transactions
        """
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")

        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

        all_results = []

        while True:
            data = self._get(endpoint, {"limit": limit, "sort": "asc"})

            if data.get("resultsCount", 0) == 0:
                break

            results = data.get("results", [])
            all_results.extend(results)

            logger.info(f"Fetched {len(results)} bars for {symbol} ({len(all_results)} total)")

            # Check if there's more data
            if len(results) < limit or "next_url" not in data:
                break

            # Get next page
            endpoint = data["next_url"].replace(self.BASE_URL, "")

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)

        # Rename columns
        column_map = {
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "transactions",
        }
        df = df.rename(columns=column_map)

        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = symbol

        # Reorder columns
        cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "transactions"]
        df = df[[c for c in cols if c in df.columns]]

        return df

    def get_daily_bars(
        self,
        symbol: str,
        years: int = 2,
    ) -> pd.DataFrame:
        """Convenience method for fetching daily bars."""
        from_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        return self.get_aggregates(symbol, multiplier=1, timespan="day", from_date=from_date)

    def get_minute_bars(
        self,
        symbol: str,
        days: int = 30,
    ) -> pd.DataFrame:
        """Convenience method for fetching 1-minute bars (requires paid plan)."""
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self.get_aggregates(symbol, multiplier=1, timespan="minute", from_date=from_date)

    def get_ticker_details(self, symbol: str) -> dict:
        """Get details about a ticker."""
        data = self._get(f"/v3/reference/tickers/{symbol}")
        return data.get("results", {})

    def get_market_status(self) -> dict:
        """Get current market status."""
        return self._get("/v1/marketstatus/now")

    def close(self):
        """Close the HTTP client."""
        self.client.close()


def download_training_data(
    symbols: list[str],
    output_dir: str = "data",
    timespan: str = "day",
    years: int = 2,
) -> dict[str, pd.DataFrame]:
    """
    Download historical data for multiple symbols and save to parquet files.

    Args:
        symbols: List of stock tickers
        output_dir: Directory to save parquet files
        timespan: day, hour, or minute
        years: Years of history to fetch

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fetcher = PolygonDataFetcher()
    results = {}

    try:
        for symbol in symbols:
            logger.info(f"Fetching {symbol}...")

            try:
                if timespan == "day":
                    df = fetcher.get_daily_bars(symbol, years=years)
                elif timespan == "minute":
                    df = fetcher.get_minute_bars(symbol, days=min(years * 365, 30))
                else:
                    df = fetcher.get_aggregates(symbol, timespan=timespan)

                if df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue

                # Save to parquet
                filename = output_path / f"{symbol}_{timespan}.parquet"
                df.to_parquet(filename, index=False)
                logger.info(f"Saved {len(df)} rows to {filename}")

                results[symbol] = df

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue

    finally:
        fetcher.close()

    return results


def load_training_data(
    symbols: list[str],
    data_dir: str = "data",
    timespan: str = "day",
) -> pd.DataFrame:
    """
    Load previously downloaded data from parquet files.

    Returns combined DataFrame with all symbols.
    """
    data_path = Path(data_dir)
    dfs = []

    for symbol in symbols:
        filename = data_path / f"{symbol}_{timespan}.parquet"
        if filename.exists():
            df = pd.read_parquet(filename)
            dfs.append(df)
            logger.info(f"Loaded {len(df)} rows for {symbol}")
        else:
            logger.warning(f"No data file for {symbol}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: Download data for top tech stocks
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"]

    print("Downloading training data...")
    results = download_training_data(
        symbols=symbols,
        output_dir="d:/Vibe_Projects/The Trader/training/data",
        timespan="day",
        years=2,
    )

    print(f"\nDownloaded data for {len(results)} symbols")
    for symbol, df in results.items():
        print(f"  {symbol}: {len(df)} bars, {df['timestamp'].min()} to {df['timestamp'].max()}")
