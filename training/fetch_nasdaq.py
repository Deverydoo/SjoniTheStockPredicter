"""
Fetch all NASDAQ stocks for training.

This script:
1. Gets the full NASDAQ listing (~4000 symbols)
2. Filters for liquid stocks with sufficient history
3. Downloads 5 years of daily data
4. Saves to parquet files
"""

import os
import sys
import logging
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fix OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = Path(__file__).parent / "data"
YEARS = 5
MAX_WORKERS = 10      # Parallel download threads

# Stock tiers for different trading strategies
STOCK_TIERS = {
    "blue_chip": {
        "min_volume": 500_000,
        "min_price": 20.0,
        "min_history_days": 1000,
        "description": "Large, established companies - low risk, steady patterns"
    },
    "mid_cap": {
        "min_volume": 100_000,
        "min_price": 5.0,
        "min_history_days": 500,
        "description": "Mid-sized companies - moderate risk, good liquidity"
    },
    "penny_stocks": {
        "min_volume": 50_000,
        "min_price": 0.01,
        "max_price": 5.0,
        "min_history_days": 100,
        "description": "Under $5 stocks - high risk, high reward potential (INO, etc.)"
    },
    "new_listings": {
        "min_volume": 25_000,
        "min_price": 0.01,
        "min_history_days": 20,  # Just 1 month minimum
        "max_history_days": 500,  # Less than 2 years
        "description": "Recent IPOs and new listings - explosive potential, unpredictable"
    },
}

# Default tier to fetch (can be overridden via command line)
DEFAULT_TIER = "all"  # Fetch all tiers


def get_nasdaq_symbols() -> list[str]:
    """Fetch all NASDAQ-listed symbols."""
    logger.info("Fetching NASDAQ symbol list...")

    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&exchange=nasdaq&download=true"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        data = resp.json()
        rows = data["data"]["rows"]

        symbols = []
        for row in rows:
            symbol = row["symbol"]
            # Filter out warrants, units, preferreds (contain non-alpha chars)
            if symbol.isalpha() and len(symbol) <= 5:
                symbols.append(symbol)

        logger.info(f"Found {len(symbols)} NASDAQ symbols")
        return sorted(symbols)

    except Exception as e:
        logger.error(f"Failed to fetch symbol list: {e}")
        return []


def classify_stock(df: pd.DataFrame) -> str:
    """
    Classify a stock into a tier based on its characteristics.
    Returns the tier name.
    """
    if df.empty:
        return None

    avg_volume = df["Volume"].mean()
    avg_price = df["Close"].mean()
    current_price = df["Close"].iloc[-1]
    history_days = len(df)

    # Check new listings first (recent IPOs)
    if history_days <= STOCK_TIERS["new_listings"].get("max_history_days", 500):
        if avg_volume >= STOCK_TIERS["new_listings"]["min_volume"]:
            return "new_listings"

    # Check penny stocks (under $5)
    if current_price < STOCK_TIERS["penny_stocks"].get("max_price", 5.0):
        if avg_volume >= STOCK_TIERS["penny_stocks"]["min_volume"]:
            if history_days >= STOCK_TIERS["penny_stocks"]["min_history_days"]:
                return "penny_stocks"

    # Check blue chip
    if (avg_price >= STOCK_TIERS["blue_chip"]["min_price"] and
        avg_volume >= STOCK_TIERS["blue_chip"]["min_volume"] and
        history_days >= STOCK_TIERS["blue_chip"]["min_history_days"]):
        return "blue_chip"

    # Check mid cap
    if (avg_price >= STOCK_TIERS["mid_cap"]["min_price"] and
        avg_volume >= STOCK_TIERS["mid_cap"]["min_volume"] and
        history_days >= STOCK_TIERS["mid_cap"]["min_history_days"]):
        return "mid_cap"

    return None  # Doesn't fit any tier


def fetch_symbol(symbol: str, years: int = 5) -> tuple[str, pd.DataFrame | None, str, str | None, dict]:
    """
    Fetch data for a single symbol.
    Returns (symbol, dataframe, status_message, tier, metadata)
    """
    metadata = {}
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{years}y", auto_adjust=True)

        if df.empty:
            return symbol, None, "no data", None, metadata

        avg_volume = df["Volume"].mean()
        avg_price = df["Close"].mean()
        current_price = df["Close"].iloc[-1]
        history_days = len(df)

        # Check for dividends (metadata flag, not a tier)
        try:
            dividends = ticker.dividends
            has_dividend = len(dividends) > 0 and dividends.sum() > 0
            if has_dividend:
                # Calculate approximate annual yield
                recent_dividends = dividends.last("1Y").sum() if len(dividends) > 0 else 0
                dividend_yield = (recent_dividends / current_price * 100) if current_price > 0 else 0
                metadata["has_dividend"] = True
                metadata["dividend_yield"] = round(dividend_yield, 2)
            else:
                metadata["has_dividend"] = False
                metadata["dividend_yield"] = 0.0
        except Exception:
            metadata["has_dividend"] = False
            metadata["dividend_yield"] = 0.0

        # Classify the stock
        tier = classify_stock(df)

        if tier is None:
            return symbol, None, f"no tier match (price=${current_price:.2f}, vol={avg_volume:,.0f}, days={history_days})", None, metadata

        # Format dataframe
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df["symbol"] = symbol
        df["tier"] = tier  # Add tier to data
        df["has_dividend"] = metadata["has_dividend"]  # Add dividend flag to each row

        # Remove timezone
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        cols = ["timestamp", "symbol", "tier", "has_dividend", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]

        div_str = f", div={metadata['dividend_yield']:.1f}%" if metadata["has_dividend"] else ""
        return symbol, df, f"{tier}: ${current_price:.2f}, {history_days} days, vol {avg_volume:,.0f}{div_str}", tier, metadata

    except Exception as e:
        return symbol, None, f"error: {str(e)[:50]}", None, metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NASDAQ data by tier")
    parser.add_argument("--tier", choices=list(STOCK_TIERS.keys()) + ["all"],
                        default="all", help="Stock tier to fetch")
    parser.add_argument("--list-tiers", action="store_true", help="List available tiers")
    args = parser.parse_args()

    if args.list_tiers:
        print("\nAvailable Stock Tiers:")
        print("=" * 60)
        for name, config in STOCK_TIERS.items():
            print(f"\n{name}:")
            print(f"  {config['description']}")
            print(f"  Min Volume: {config['min_volume']:,}")
            print(f"  Min Price: ${config['min_price']}")
            if 'max_price' in config:
                print(f"  Max Price: ${config['max_price']}")
            print(f"  Min History: {config['min_history_days']} days")
            if 'max_history_days' in config:
                print(f"  Max History: {config['max_history_days']} days")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create tier subdirectories
    for tier_name in STOCK_TIERS.keys():
        (OUTPUT_DIR / tier_name).mkdir(parents=True, exist_ok=True)

    # Get all NASDAQ symbols
    all_symbols = get_nasdaq_symbols()

    if not all_symbols:
        logger.error("No symbols found!")
        return

    logger.info(f"Starting download of {len(all_symbols)} symbols with {MAX_WORKERS} workers...")
    logger.info(f"Fetching tier: {args.tier}")
    print("\nStock Tiers:")
    for name, config in STOCK_TIERS.items():
        print(f"  {name}: {config['description']}")

    # Track by tier
    tier_counts = {tier: [] for tier in STOCK_TIERS.keys()}
    dividend_stocks = []  # Track dividend payers separately
    symbol_metadata = {}  # Store metadata for all symbols
    failed = []
    no_tier = []

    start_time = time.time()

    # Download in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_symbol, sym, YEARS): sym for sym in all_symbols}

        for i, future in enumerate(as_completed(futures)):
            symbol, df, status, tier, meta = future.result()

            if df is not None and tier is not None:
                # Skip if not the requested tier (unless fetching all)
                if args.tier != "all" and tier != args.tier:
                    continue

                # Save to tier-specific directory AND main directory
                tier_filepath = OUTPUT_DIR / tier / f"{symbol}_day.parquet"
                main_filepath = OUTPUT_DIR / f"{symbol}_day.parquet"
                df.to_parquet(tier_filepath, index=False)
                df.to_parquet(main_filepath, index=False)

                tier_counts[tier].append(symbol)
                symbol_metadata[symbol] = {"tier": tier, **meta}

                # Track dividend stocks
                if meta.get("has_dividend"):
                    dividend_stocks.append((symbol, meta.get("dividend_yield", 0)))

                logger.info(f"[{i+1}/{len(all_symbols)}] {symbol}: {status}")

            elif "error" in status:
                failed.append((symbol, status))
            else:
                no_tier.append((symbol, status))

            # Progress update every 100 symbols
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(all_symbols) - i - 1) / rate / 60
                total_saved = sum(len(v) for v in tier_counts.values())
                div_count = len(dividend_stocks)
                logger.info(f"Progress: {i+1}/{len(all_symbols)} ({total_saved} saved, {div_count} dividends) - ETA: {remaining:.1f} min")

    elapsed = time.time() - start_time
    total_saved = sum(len(v) for v in tier_counts.values())

    # Summary
    print("\n" + "=" * 70)
    print("NASDAQ DATA DOWNLOAD COMPLETE - TIERED SYSTEM")
    print("=" * 70)
    print(f"Total symbols scanned: {len(all_symbols)}")
    print(f"Total saved:           {total_saved}")
    print(f"Dividend payers:       {len(dividend_stocks)}")
    print(f"No tier match:         {len(no_tier)}")
    print(f"Errors:                {len(failed)}")
    print(f"Time elapsed:          {elapsed/60:.1f} minutes")
    print("\n" + "-" * 70)
    print("BREAKDOWN BY TIER:")
    print("-" * 70)
    for tier_name, symbols in tier_counts.items():
        config = STOCK_TIERS[tier_name]
        # Count dividend payers in this tier
        tier_div_count = sum(1 for s in symbols if symbol_metadata.get(s, {}).get("has_dividend"))
        print(f"\n{tier_name.upper()} ({len(symbols)} stocks, {tier_div_count} dividend payers)")
        print(f"  {config['description']}")
        print(f"  Saved to: {OUTPUT_DIR / tier_name}")
        if symbols:
            print(f"  Examples: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")

    # Show top dividend payers
    if dividend_stocks:
        print("\n" + "-" * 70)
        print("TOP DIVIDEND PAYERS:")
        print("-" * 70)
        top_div = sorted(dividend_stocks, key=lambda x: x[1], reverse=True)[:10]
        for sym, yield_pct in top_div:
            tier = symbol_metadata.get(sym, {}).get("tier", "unknown")
            print(f"  {sym}: {yield_pct:.1f}% yield ({tier})")

    print("\n" + "=" * 70)

    # Save symbol lists per tier
    for tier_name, symbols in tier_counts.items():
        if symbols:
            with open(OUTPUT_DIR / tier_name / "symbols.txt", "w") as f:
                f.write("\n".join(sorted(symbols)))
            logger.info(f"Saved {len(symbols)} {tier_name} symbols")

    # Save combined list
    all_saved = []
    for symbols in tier_counts.values():
        all_saved.extend(symbols)
    with open(OUTPUT_DIR / "all_symbols.txt", "w") as f:
        f.write("\n".join(sorted(all_saved)))

    # Save tier metadata
    import json
    metadata = {
        "tiers": STOCK_TIERS,
        "counts": {k: len(v) for k, v in tier_counts.items()},
        "symbols": {k: sorted(v) for k, v in tier_counts.items()},
        "dividend_count": len(dividend_stocks),
        "dividend_stocks": [s for s, _ in sorted(dividend_stocks, key=lambda x: x[1], reverse=True)],
        "symbol_metadata": symbol_metadata,  # Full metadata for each symbol
        "fetch_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUTPUT_DIR / "tier_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nData saved to: {OUTPUT_DIR}")
    print("  - Main directory: all stocks (flat)")
    print("  - Subdirectories: organized by tier")
    print("  - tier_metadata.json: tier definitions and symbol lists")


if __name__ == "__main__":
    main()
