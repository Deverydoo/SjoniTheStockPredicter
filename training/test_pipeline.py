#!/usr/bin/env python3
"""Test script to validate the training pipeline components."""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gdelt_historical import PriceDerivedSentiment
from institutional_flow import InstitutionalFlowAnalyzer
from sector_classifier import SectorClassifier
from features import add_technical_indicators

def test_data_loading():
    """Test that we can load the historical data."""
    print("=" * 60)
    print("1. DATA LOADING TEST")
    print("=" * 60)

    data_dir = Path("data/historical")
    files = list(data_dir.glob("*.parquet"))
    print(f"Total parquet files: {len(files)}")

    if len(files) == 0:
        print("ERROR: No parquet files found!")
        return None, None

    # Find files with significant history (>1000 days)
    long_history = []
    for f in files:
        df = pd.read_parquet(f)
        if len(df) > 1000:
            long_history.append((f.stem.replace("_historical", ""), len(df)))

    long_history.sort(key=lambda x: -x[1])
    print(f"\nStocks with >1000 days of history: {len(long_history)}")
    print(f"Top 10 by history length:")
    for sym, days in long_history[:10]:
        print(f"  {sym}: {days} days")

    # Load SPY for market reference
    spy_file = data_dir / "SPY_historical.parquet"
    if spy_file.exists():
        spy_df = pd.read_parquet(spy_file)
        print(f"\nSPY data: {len(spy_df)} days ({spy_df.index.min().date()} to {spy_df.index.max().date()})")
    else:
        print("\nWARNING: SPY data not found!")
        spy_df = None

    # Load AAPL as test stock
    aapl_file = data_dir / "AAPL_historical.parquet"
    if aapl_file.exists():
        aapl_df = pd.read_parquet(aapl_file)
        print(f"AAPL data: {len(aapl_df)} days ({aapl_df.index.min().date()} to {aapl_df.index.max().date()})")
    else:
        print("WARNING: AAPL data not found!")
        aapl_df = None

    return spy_df, aapl_df


def test_sentiment_pipeline(stock_df, market_df):
    """Test the price-derived sentiment calculator."""
    print("\n" + "=" * 60)
    print("2. SENTIMENT PIPELINE TEST")
    print("=" * 60)

    if stock_df is None:
        print("Skipping - no stock data")
        return None

    sentiment = PriceDerivedSentiment()

    # Calculate sentiment
    sent_features = sentiment.calculate_sentiment_proxy(stock_df, market_df)

    print(f"Sentiment features shape: {sent_features.shape}")
    print(f"Columns: {list(sent_features.columns)}")

    # Check quality
    nan_pct = sent_features.isna().sum() / len(sent_features) * 100
    print(f"\nNaN percentage by column:")
    for col, pct in nan_pct.items():
        status = "OK" if pct < 5 else "WARN" if pct < 20 else "ERROR"
        print(f"  {col}: {pct:.1f}% [{status}]")

    # Show sample values
    cols = ['overnight_gap', 'close_position', 'sentiment_proxy', 'sentiment_proxy_smooth']
    cols = [c for c in cols if c in sent_features.columns]
    print(f"\nSample values (last 5 rows):")
    print(sent_features[cols].tail().to_string())

    # Statistics
    print(f"\nSentiment proxy statistics:")
    print(f"  Mean: {sent_features['sentiment_proxy'].mean():.4f}")
    print(f"  Std:  {sent_features['sentiment_proxy'].std():.4f}")
    print(f"  Min:  {sent_features['sentiment_proxy'].min():.4f}")
    print(f"  Max:  {sent_features['sentiment_proxy'].max():.4f}")

    return sent_features


def test_institutional_flow(stock_df):
    """Test the institutional flow analyzer."""
    print("\n" + "=" * 60)
    print("3. INSTITUTIONAL FLOW TEST")
    print("=" * 60)

    if stock_df is None:
        print("Skipping - no stock data")
        return None

    analyzer = InstitutionalFlowAnalyzer()

    # Analyze
    inst_features = analyzer.analyze(stock_df)

    print(f"Institutional features shape: {inst_features.shape}")
    print(f"Columns: {list(inst_features.columns)}")

    # Check quality
    nan_pct = inst_features.isna().sum() / len(inst_features) * 100
    print(f"\nNaN percentage by column:")
    for col, pct in nan_pct.items():
        status = "OK" if pct < 5 else "WARN" if pct < 20 else "ERROR"
        print(f"  {col}: {pct:.1f}% [{status}]")

    # Show sample values
    cols = ['smart_money_flow', 'weak_hands_score', 'accumulation_distribution']
    cols = [c for c in cols if c in inst_features.columns]
    print(f"\nSample values (last 5 rows):")
    print(inst_features[cols].tail().to_string())

    return inst_features


def test_sector_classifier():
    """Test the sector classifier."""
    print("\n" + "=" * 60)
    print("4. SECTOR CLASSIFIER TEST")
    print("=" * 60)

    classifier = SectorClassifier(cache_path="data/sector_cache.json")

    # Test a few symbols
    test_symbols = ["AAPL", "NVDA", "JPM", "XOM", "TSLA"]

    print("Testing classification:")
    for sym in test_symbols:
        sector_id, industry_id, sector_name, industry_name = classifier.classify(sym)
        print(f"  {sym}: [{sector_id}] {sector_name} / [{industry_id}] {industry_name}")

    return classifier


def test_technical_indicators(stock_df):
    """Test the technical indicator calculation."""
    print("\n" + "=" * 60)
    print("5. TECHNICAL INDICATORS TEST")
    print("=" * 60)

    if stock_df is None:
        print("Skipping - no stock data")
        return None

    # Add technical indicators
    df_with_indicators = add_technical_indicators(stock_df)

    new_cols = set(df_with_indicators.columns) - set(stock_df.columns)
    print(f"New columns added: {len(new_cols)}")
    print(f"Columns: {sorted(new_cols)}")

    # Check for expected columns
    expected = ['returns', 'rsi_14', 'macd', 'bb_width', 'atr_14']
    for col in expected:
        if col in df_with_indicators.columns:
            nan_pct = df_with_indicators[col].isna().sum() / len(df_with_indicators) * 100
            print(f"  {col}: present, {nan_pct:.1f}% NaN")
        else:
            print(f"  {col}: MISSING")

    return df_with_indicators


def test_full_pipeline():
    """Test the full data processing pipeline."""
    print("\n" + "=" * 60)
    print("6. FULL PIPELINE INTEGRATION TEST")
    print("=" * 60)

    data_dir = Path("data/historical")

    # Load data
    spy_df = pd.read_parquet(data_dir / "SPY_historical.parquet") if (data_dir / "SPY_historical.parquet").exists() else None

    # Find a stock with long history
    for f in sorted(data_dir.glob("*.parquet"), key=lambda x: x.stat().st_size, reverse=True)[:20]:
        df = pd.read_parquet(f)
        if len(df) > 2000:  # Need at least ~8 years for good training
            symbol = f.stem.replace("_historical", "")
            print(f"Testing with {symbol} ({len(df)} days)")
            break
    else:
        print("No suitable long-history stock found!")
        return False

    # Initialize processors
    sentiment = PriceDerivedSentiment()
    inst_analyzer = InstitutionalFlowAnalyzer()
    sector_classifier = SectorClassifier(cache_path="data/sector_cache.json")

    # Process
    print(f"\nProcessing {symbol}...")

    # 1. Technical indicators
    df = add_technical_indicators(df)
    print(f"  After technical indicators: {df.shape}")

    # 2. Sentiment
    sent_features = sentiment.calculate_sentiment_proxy(df, spy_df)
    df = pd.concat([df, sent_features], axis=1)
    print(f"  After sentiment: {df.shape}")

    # 3. Institutional flow
    inst_features = inst_analyzer.analyze(df)
    df = pd.concat([df, inst_features], axis=1)
    print(f"  After institutional: {df.shape}")

    # 4. Sector classification
    sector_id, industry_id, sector_name, industry_name = sector_classifier.classify(symbol)
    print(f"  Sector: {sector_name} ({sector_id}), Industry: {industry_name} ({industry_id})")

    # 5. Drop NaN and check remaining
    df_clean = df.dropna()
    print(f"  After dropping NaN: {df_clean.shape} ({len(df_clean)/len(df)*100:.1f}% retained)")

    # 6. Check if enough data for training
    sequence_length = 252
    prediction_horizon = 5
    min_required = sequence_length + prediction_horizon + 50

    if len(df_clean) >= min_required:
        print(f"  ✓ Sufficient data for training ({len(df_clean)} >= {min_required})")
        return True
    else:
        print(f"  ✗ Insufficient data ({len(df_clean)} < {min_required})")
        return False


def main():
    print("Sjoni Training Pipeline Validation")
    print("=" * 60)

    # Run all tests
    spy_df, aapl_df = test_data_loading()
    test_sentiment_pipeline(aapl_df, spy_df)
    test_institutional_flow(aapl_df)
    test_sector_classifier()
    test_technical_indicators(aapl_df)
    success = test_full_pipeline()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if success:
        print("✓ Pipeline validation PASSED")
        print("\nReady to pre-format sentiment data and begin training!")
    else:
        print("✗ Pipeline validation FAILED - check errors above")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
