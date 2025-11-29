#!/usr/bin/env python3
"""Quick check of data file structure."""
import pandas as pd
from pathlib import Path

data_dir = Path("data/historical")

# Check market_context
mc_file = data_dir / "market_context_historical.parquet"
if mc_file.exists():
    df = pd.read_parquet(mc_file)
    print("market_context columns:", list(df.columns))
    print("Shape:", df.shape)
else:
    print("market_context file not found")

# Check SPY
spy_file = data_dir / "SPY_historical.parquet"
if spy_file.exists():
    df = pd.read_parquet(spy_file)
    print("\nSPY columns:", list(df.columns)[:10])
    print("SPY Shape:", df.shape)

# Check AAPL
aapl_file = data_dir / "AAPL_historical.parquet"
if aapl_file.exists():
    df = pd.read_parquet(aapl_file)
    print("\nAAPL columns:", list(df.columns)[:10])
    print("AAPL Shape:", df.shape)
