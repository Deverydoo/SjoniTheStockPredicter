#!/usr/bin/env python3
"""
Pre-format Sentiment Data for Sjoni Training

This script pre-computes:
1. Price-derived sentiment features for all stocks
2. Institutional flow features
3. Caches sector/industry classifications

Running this before training significantly speeds up data loading.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gdelt_historical import PriceDerivedSentiment
from institutional_flow import InstitutionalFlowAnalyzer
from sector_classifier import SectorClassifier
from features import add_technical_indicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def process_stock(
    file_path: Path,
    market_df: pd.DataFrame,
    sentiment_processor: PriceDerivedSentiment,
    inst_analyzer: InstitutionalFlowAnalyzer,
    output_dir: Path
) -> dict:
    """Process a single stock file and save enhanced features."""
    symbol = file_path.stem.replace("_historical", "")

    try:
        # Load data
        df = pd.read_parquet(file_path)

        # Skip market_context file
        if 'market_context' in symbol.lower():
            return {"symbol": symbol, "status": "skipped", "reason": "market_context"}

        # Skip if too short
        if len(df) < 300:  # Need at least ~1 year + warmup
            return {"symbol": symbol, "status": "skipped", "reason": f"too_short ({len(df)} rows)"}

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {"symbol": symbol, "status": "error", "reason": f"missing columns: {missing}"}

        # 1. Calculate sentiment features
        sent_features = sentiment_processor.calculate_sentiment_proxy(df, market_df)

        # 2. Calculate institutional flow features
        inst_features = inst_analyzer.analyze(df)

        # 3. Ensure we have technical indicators (add if missing)
        tech_cols = ['returns', 'rsi_14', 'macd', 'bb_width', 'atr_14']
        if not all(c in df.columns for c in tech_cols):
            df = add_technical_indicators(df)

        # 4. Combine all features
        enhanced_df = pd.concat([df, sent_features, inst_features], axis=1)

        # 5. Remove duplicate columns
        enhanced_df = enhanced_df.loc[:, ~enhanced_df.columns.duplicated()]

        # 6. Save enhanced data
        output_file = output_dir / f"{symbol}_enhanced.parquet"
        enhanced_df.to_parquet(output_file)

        # Calculate stats
        nan_pct = enhanced_df.isna().sum().sum() / enhanced_df.size * 100

        return {
            "symbol": symbol,
            "status": "success",
            "rows": len(enhanced_df),
            "cols": len(enhanced_df.columns),
            "nan_pct": round(nan_pct, 2)
        }

    except Exception as e:
        return {"symbol": symbol, "status": "error", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Pre-format sentiment data for training")
    parser.add_argument('--data-dir', type=str, default='data/historical',
                       help='Input data directory')
    parser.add_argument('--output-dir', type=str, default='data/enhanced',
                       help='Output directory for enhanced data')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of files to process (for testing)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all parquet files
    files = list(data_dir.glob("*_historical.parquet"))

    if args.limit:
        files = files[:args.limit]

    logger.info(f"Found {len(files)} parquet files in {data_dir}")

    # Load market reference (SPY)
    spy_file = data_dir / "SPY_historical.parquet"
    if spy_file.exists():
        market_df = pd.read_parquet(spy_file)
        logger.info(f"Loaded SPY market reference: {len(market_df)} rows")
    else:
        logger.warning("SPY not found, using None for market reference")
        market_df = None

    # Initialize processors (shared across threads)
    sentiment_processor = PriceDerivedSentiment()
    inst_analyzer = InstitutionalFlowAnalyzer()

    # Process files
    logger.info(f"Processing {len(files)} files with {args.workers} workers...")

    results = []
    success_count = 0
    error_count = 0
    skipped_count = 0

    # Use sequential processing for stability
    for file_path in tqdm(files, desc="Processing"):
        result = process_stock(
            file_path,
            market_df,
            sentiment_processor,
            inst_analyzer,
            output_dir
        )
        results.append(result)

        if result["status"] == "success":
            success_count += 1
        elif result["status"] == "error":
            error_count += 1
            logger.warning(f"Error processing {result['symbol']}: {result.get('reason', 'unknown')}")
        else:
            skipped_count += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PRE-FORMATTING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total files:  {len(files)}")
    logger.info(f"Success:      {success_count}")
    logger.info(f"Skipped:      {skipped_count}")
    logger.info(f"Errors:       {error_count}")
    logger.info(f"Output dir:   {output_dir}")

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_file = output_dir / "processing_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary saved to: {summary_file}")

    # Show success stats
    success_results = [r for r in results if r["status"] == "success"]
    if success_results:
        avg_rows = np.mean([r["rows"] for r in success_results])
        avg_cols = np.mean([r["cols"] for r in success_results])
        avg_nan = np.mean([r["nan_pct"] for r in success_results])

        logger.info(f"\nSuccess statistics:")
        logger.info(f"  Average rows: {avg_rows:.0f}")
        logger.info(f"  Average cols: {avg_cols:.0f}")
        logger.info(f"  Average NaN%: {avg_nan:.2f}%")

    return success_count > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
