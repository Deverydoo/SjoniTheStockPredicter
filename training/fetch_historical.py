"""
Extended Historical Data Fetcher

Fetches data back to 1999 for comprehensive market cycle analysis:
- Dot-com bubble (1999-2000)
- Dot-com crash (2000-2002)
- Housing bubble (2006-2007)
- Financial crisis (2008-2009)
- Recovery (2009-2019)
- COVID crash (2020)
- Meme stock era (2021)
- Tech correction (2022)
- AI boom (2023-present)

Also fetches delisted stocks to avoid survivorship bias.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
import time
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StockMetadata:
    """Metadata for each stock"""
    symbol: str
    name: str
    sector: str
    industry: str
    ipo_date: Optional[str]
    delisted: bool = False
    delisted_date: Optional[str] = None
    delisting_reason: Optional[str] = None
    has_dividend: bool = False
    dividend_yield: float = 0.0
    market_cap_tier: str = "unknown"  # mega, large, mid, small, micro, nano
    data_start_date: Optional[str] = None
    data_end_date: Optional[str] = None
    total_trading_days: int = 0


# GICS Sector mapping
GICS_SECTORS = {
    "Technology": 0,
    "Information Technology": 0,
    "Healthcare": 1,
    "Health Care": 1,
    "Financials": 2,
    "Financial Services": 2,
    "Consumer Cyclical": 3,
    "Consumer Discretionary": 3,
    "Consumer Defensive": 4,
    "Consumer Staples": 4,
    "Industrials": 5,
    "Energy": 6,
    "Utilities": 7,
    "Real Estate": 8,
    "Basic Materials": 9,
    "Materials": 9,
    "Communication Services": 10,
    "Telecommunications": 10,
}

# Notable historical stocks that got delisted (for survivorship bias correction)
HISTORICAL_DELISTED = [
    # Dot-com bubble casualties
    {"symbol": "WCOM", "name": "WorldCom", "delisted": "2002", "reason": "fraud_bankruptcy"},
    {"symbol": "ENRNQ", "name": "Enron", "delisted": "2001", "reason": "fraud_bankruptcy"},
    {"symbol": "GLW", "name": "Global Crossing", "delisted": "2002", "reason": "bankruptcy"},

    # Financial crisis casualties
    {"symbol": "LEH", "name": "Lehman Brothers", "delisted": "2008", "reason": "bankruptcy"},
    {"symbol": "BSC", "name": "Bear Stearns", "delisted": "2008", "reason": "acquired_distressed"},
    {"symbol": "WM", "name": "Washington Mutual", "delisted": "2008", "reason": "bankruptcy"},
    {"symbol": "CFC", "name": "Countrywide Financial", "delisted": "2008", "reason": "acquired_distressed"},
    {"symbol": "AIG", "name": "AIG (old)", "delisted": "2008", "reason": "restructured"},

    # Recent notable delistings
    {"symbol": "HMNY", "name": "Helios and Matheson (MoviePass)", "delisted": "2019", "reason": "bankruptcy"},
    {"symbol": "LK", "name": "Luckin Coffee", "delisted": "2020", "reason": "fraud"},
    {"symbol": "RIDE", "name": "Lordstown Motors", "delisted": "2023", "reason": "bankruptcy"},
    {"symbol": "BBBYQ", "name": "Bed Bath & Beyond", "delisted": "2023", "reason": "bankruptcy"},
    {"symbol": "SIVBQ", "name": "Silicon Valley Bank", "delisted": "2023", "reason": "bankruptcy"},
    {"symbol": "FRCB", "name": "First Republic Bank", "delisted": "2023", "reason": "acquired_distressed"},

    # SPAC failures
    {"symbol": "NKLA", "name": "Nikola (near-delisting)", "delisted": None, "reason": "fraud_ongoing"},
    {"symbol": "HYLN", "name": "Hyliion", "delisted": None, "reason": "spac_failure"},
]

# Blue chip stocks with long history (for 25-year data)
LONG_HISTORY_STOCKS = [
    # Tech giants
    "MSFT", "AAPL", "INTC", "CSCO", "ORCL", "IBM", "DELL", "HPQ",
    "TXN", "QCOM", "AMAT", "KLAC", "LRCX", "MU", "NVDA", "AMD",

    # Internet era
    "AMZN", "EBAY", "YHOO", "GOOG", "GOOGL", "META", "NFLX",

    # Telecom
    "T", "VZ", "TMUS", "CMCSA",

    # Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW",

    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "AMGN", "GILD", "BIIB",

    # Consumer
    "WMT", "COST", "TGT", "HD", "LOW", "SBUX", "MCD", "NKE", "DIS",

    # Industrial
    "GE", "CAT", "BA", "HON", "MMM", "UPS", "FDX",

    # Energy
    "XOM", "CVX", "COP", "SLB", "HAL",

    # Index ETFs (for market context)
    "SPY", "QQQ", "DIA", "IWM", "VTI",
]


def get_sector_id(sector_name: str) -> int:
    """Convert sector name to ID"""
    if not sector_name:
        return -1
    return GICS_SECTORS.get(sector_name, -1)


def get_market_cap_tier(market_cap: float) -> str:
    """Classify by market cap"""
    if market_cap is None or market_cap <= 0:
        return "unknown"
    if market_cap >= 200e9:
        return "mega"  # >$200B
    elif market_cap >= 10e9:
        return "large"  # $10-200B
    elif market_cap >= 2e9:
        return "mid"  # $2-10B
    elif market_cap >= 300e6:
        return "small"  # $300M-2B
    elif market_cap >= 50e6:
        return "micro"  # $50-300M
    else:
        return "nano"  # <$50M


def fetch_extended_history(
    symbol: str,
    start_date: str = "1999-01-01",
    end_date: str = None
) -> Tuple[str, pd.DataFrame, StockMetadata]:
    """
    Fetch extended historical data for a symbol

    Returns: (symbol, dataframe, metadata)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    metadata = StockMetadata(
        symbol=symbol,
        name="",
        sector="",
        industry="",
        ipo_date=None
    )

    try:
        ticker = yf.Ticker(symbol)

        # Get stock info
        try:
            info = ticker.info
            metadata.name = info.get('longName', info.get('shortName', symbol))
            metadata.sector = info.get('sector', '')
            metadata.industry = info.get('industry', '')

            # Market cap tier
            market_cap = info.get('marketCap', 0)
            metadata.market_cap_tier = get_market_cap_tier(market_cap)

            # Dividend info
            div_yield = info.get('dividendYield', 0) or 0
            metadata.has_dividend = div_yield > 0
            metadata.dividend_yield = round(div_yield * 100, 2)

        except Exception as e:
            logger.debug(f"{symbol}: Could not get info - {e}")

        # Fetch historical data
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

        if df.empty:
            logger.warning(f"{symbol}: No data available")
            return symbol, None, metadata

        # Clean column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        # Add symbol column
        df['symbol'] = symbol

        # Calculate additional features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']

        # Price relative to moving averages
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['price_to_sma50'] = df['close'] / df['sma_50']
        df['price_to_sma200'] = df['close'] / df['sma_200']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()

        # Market regime indicators
        df['trend'] = np.where(df['close'] > df['sma_200'], 1, -1)
        df['momentum'] = df['close'].pct_change(20)  # 20-day momentum

        # Days since listing (for IPO detection)
        df['days_listed'] = range(len(df))

        # Sector ID
        df['sector_id'] = get_sector_id(metadata.sector)

        # Update metadata
        metadata.data_start_date = df.index[0].strftime("%Y-%m-%d")
        metadata.data_end_date = df.index[-1].strftime("%Y-%m-%d")
        metadata.total_trading_days = len(df)

        # Detect if this might be delisted (no recent data)
        last_date = df.index[-1]
        days_since_last = (datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days
        if days_since_last > 30:  # No data in last 30 days
            metadata.delisted = True
            metadata.delisted_date = df.index[-1].strftime("%Y-%m-%d")

        logger.info(
            f"{symbol}: {len(df)} days from {metadata.data_start_date} to {metadata.data_end_date} "
            f"({metadata.sector or 'Unknown sector'})"
        )

        return symbol, df, metadata

    except Exception as e:
        logger.error(f"{symbol}: Error - {e}")
        return symbol, None, metadata


def fetch_all_nasdaq_symbols() -> List[str]:
    """Fetch current NASDAQ symbols"""
    try:
        url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
        df = pd.read_csv(url, sep='|')
        symbols = df['Symbol'].dropna().tolist()
        # Remove test symbols
        symbols = [s for s in symbols if not s.endswith('test') and len(s) <= 5]
        return symbols[:-1]  # Remove last row (file trailer)
    except Exception as e:
        logger.error(f"Failed to fetch NASDAQ symbols: {e}")
        # Fallback to saved list
        symbol_file = Path("data/nasdaq_symbols.txt")
        if symbol_file.exists():
            with open(symbol_file) as f:
                return [line.strip() for line in f if line.strip()]
        return []


def fetch_historical_data_parallel(
    symbols: List[str],
    start_date: str = "1999-01-01",
    max_workers: int = 10,
    output_dir: str = "data/historical"
) -> Dict[str, StockMetadata]:
    """
    Fetch historical data for multiple symbols in parallel
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_dict = {}
    successful = 0
    failed = 0

    logger.info(f"Fetching {len(symbols)} symbols with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_extended_history, symbol, start_date): symbol
            for symbol in symbols
        }

        for i, future in enumerate(as_completed(futures)):
            symbol = futures[future]

            try:
                sym, df, metadata = future.result()

                if df is not None and len(df) > 100:  # At least 100 days of data
                    # Save parquet
                    parquet_path = output_path / f"{symbol}_historical.parquet"
                    df.to_parquet(parquet_path)

                    metadata_dict[symbol] = asdict(metadata)
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"{symbol}: Exception - {e}")
                failed += 1

            # Progress update
            if (i + 1) % 50 == 0:
                logger.info(
                    f"Progress: {i + 1}/{len(symbols)} "
                    f"(success: {successful}, failed: {failed})"
                )

            # Rate limiting
            time.sleep(0.1)

    # Save metadata
    metadata_path = output_path / "stock_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)

    logger.info(f"\nCompleted: {successful} successful, {failed} failed")
    logger.info(f"Data saved to {output_path}")

    return metadata_dict


def add_market_context_data(output_dir: str = "data/historical"):
    """
    Add market-wide context data (VIX, sector indices, etc.)
    """
    output_path = Path(output_dir)

    # Market indices and context
    context_symbols = {
        "^VIX": "vix",           # Volatility Index
        "^GSPC": "sp500",        # S&P 500
        "^IXIC": "nasdaq",       # NASDAQ Composite
        "^DJI": "dow",           # Dow Jones
        "^RUT": "russell2000",   # Russell 2000
        "^TNX": "treasury_10y",  # 10-Year Treasury Yield
        "GC=F": "gold",          # Gold Futures
        "CL=F": "oil",           # Crude Oil Futures
        "DX-Y.NYB": "dxy",       # US Dollar Index
    }

    market_data = {}

    for symbol, name in context_symbols.items():
        logger.info(f"Fetching market context: {name}")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start="1999-01-01", auto_adjust=True)

            if not df.empty:
                df.columns = [f"{name}_{c.lower()}" for c in df.columns]
                market_data[name] = df
                logger.info(f"  {name}: {len(df)} days")

        except Exception as e:
            logger.error(f"Failed to fetch {name}: {e}")

    # Merge all market data
    if market_data:
        combined = None
        for name, df in market_data.items():
            if combined is None:
                combined = df
            else:
                combined = combined.join(df, how='outer')

        # Forward fill missing values
        combined = combined.fillna(method='ffill')

        # Save
        combined.to_parquet(output_path / "market_context.parquet")
        logger.info(f"Market context saved: {len(combined)} days")

    return combined


def generate_training_features(
    symbol: str,
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    sentiment_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Combine stock data with market context and sentiment
    for model training
    """
    # Merge stock with market context
    df = stock_df.join(market_df, how='left')

    # Add sentiment if available
    if sentiment_df is not None and not sentiment_df.empty:
        df = df.join(sentiment_df, how='left')
        # Fill missing sentiment with neutral
        sentiment_cols = [c for c in df.columns if 'sentiment' in c.lower()]
        for col in sentiment_cols:
            df[col] = df[col].fillna(0)

    # Forward fill market data
    market_cols = [c for c in df.columns if any(
        name in c for name in ['vix', 'sp500', 'nasdaq', 'dow', 'treasury', 'gold', 'oil', 'dxy']
    )]
    df[market_cols] = df[market_cols].fillna(method='ffill')

    return df


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Fetch extended historical data')
    parser.add_argument('--symbols', type=str, default='all',
                        help='Symbols to fetch: "all", "bluechip", or comma-separated list')
    parser.add_argument('--start', type=str, default='1999-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel workers')
    parser.add_argument('--output', type=str, default='data/historical',
                        help='Output directory')
    parser.add_argument('--market-context', action='store_true',
                        help='Also fetch market context data (VIX, indices, etc.)')

    args = parser.parse_args()

    # Determine symbols to fetch
    if args.symbols == 'all':
        symbols = fetch_all_nasdaq_symbols()
        logger.info(f"Fetching all {len(symbols)} NASDAQ symbols")
    elif args.symbols == 'bluechip':
        symbols = LONG_HISTORY_STOCKS
        logger.info(f"Fetching {len(symbols)} blue chip stocks with long history")
    else:
        symbols = [s.strip() for s in args.symbols.split(',')]
        logger.info(f"Fetching {len(symbols)} specified symbols")

    # Fetch stock data
    metadata = fetch_historical_data_parallel(
        symbols,
        start_date=args.start,
        max_workers=args.workers,
        output_dir=args.output
    )

    # Fetch market context
    if args.market_context:
        add_market_context_data(args.output)

    # Summary statistics
    if metadata:
        sectors = {}
        tiers = {}
        years_of_data = []

        for sym, meta in metadata.items():
            sector = meta.get('sector', 'Unknown')
            tier = meta.get('market_cap_tier', 'unknown')
            days = meta.get('total_trading_days', 0)

            sectors[sector] = sectors.get(sector, 0) + 1
            tiers[tier] = tiers.get(tier, 0) + 1
            years_of_data.append(days / 252)  # Approximate years

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)

        print(f"\nTotal symbols: {len(metadata)}")
        print(f"Average history: {np.mean(years_of_data):.1f} years")
        print(f"Max history: {np.max(years_of_data):.1f} years")

        print("\nBy Sector:")
        for sector, count in sorted(sectors.items(), key=lambda x: -x[1]):
            print(f"  {sector}: {count}")

        print("\nBy Market Cap Tier:")
        for tier, count in sorted(tiers.items(), key=lambda x: -x[1]):
            print(f"  {tier}: {count}")


if __name__ == "__main__":
    main()
