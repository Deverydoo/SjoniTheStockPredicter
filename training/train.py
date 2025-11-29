"""
Main training script for ArgusTrader ML models.

Usage:
    python train.py --download              # Download data first
    python train.py --train                 # Train the model
    python train.py --download --train      # Both
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Fix OpenMP duplicate library issue on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_fetcher import download_training_data, download_yahoo_data, load_training_data
from features import (
    add_technical_indicators,
    create_target_variables,
    get_feature_columns,
    normalize_features,
    prepare_sequences,
)
from models import create_model
from trainer import Trainer, create_data_loaders, evaluate_predictions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Configuration
CONFIG = {
    # Symbols to train on - set to None to auto-detect from data directory
    "symbols": None,  # Will load all *_day.parquet files

    # Tier-based training (None = all tiers, or specify: blue_chip, mid_cap, penny_stocks, new_listings)
    "tier": None,  # Train on all tiers by default

    # Data settings
    "data_dir": "data",
    "timespan": "day",
    "years": 5,  # 5 years of data from Yahoo Finance
    "data_source": "yahoo",  # "yahoo" (free, 20+ years) or "polygon" (requires API key)

    # Sequence settings
    "sequence_length": 60,  # 60 days of history
    "prediction_horizon": 5,  # Predict 5 days ahead

    # Model settings
    "model_type": "tft",  # lstm, transformer, cnn_lstm, attention_lstm, tft
    "hidden_size": 128,   # TFT default (uses more parameters internally)
    "num_layers": 8,      # Number of attention layers
    "num_heads": 4,       # Attention heads
    "dropout": 0.1,       # TFT default dropout

    # Training settings
    "batch_size": 64,    # Larger batches for more data
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "early_stopping_patience": 15,
    "train_split": 0.8,

    # Output
    "model_dir": "models",
}


def get_available_symbols(data_dir: Path, tier: str = None) -> list[str]:
    """Auto-detect symbols from parquet files in data directory.

    Args:
        data_dir: Base data directory
        tier: Optional tier to filter (blue_chip, mid_cap, penny_stocks, new_listings)
    """
    symbols = []

    if tier:
        # Look in tier-specific directory
        tier_dir = data_dir / tier
        if tier_dir.exists():
            for f in tier_dir.glob("*_day.parquet"):
                symbol = f.stem.replace("_day", "")
                symbols.append(symbol)
            logger.info(f"Loading from tier '{tier}': {len(symbols)} symbols")
        else:
            logger.warning(f"Tier directory not found: {tier_dir}")
            # Fall back to main directory
            for f in data_dir.glob("*_day.parquet"):
                symbol = f.stem.replace("_day", "")
                symbols.append(symbol)
    else:
        # Load all from main directory
        for f in data_dir.glob("*_day.parquet"):
            symbol = f.stem.replace("_day", "")
            symbols.append(symbol)

    return sorted(symbols)


def download_data():
    """Download historical data from configured source."""
    source = CONFIG.get("data_source", "yahoo")
    logger.info(f"Downloading historical data from {source}...")

    data_dir = Path(__file__).parent / CONFIG["data_dir"]

    if source == "yahoo":
        # Use Yahoo Finance (FREE, 20+ years of data)
        results = download_yahoo_data(
            symbols=CONFIG["symbols"],
            output_dir=str(data_dir),
            years=CONFIG["years"],
        )
    else:
        # Use Polygon.io (requires API key, 2 years free)
        results = download_training_data(
            symbols=CONFIG["symbols"],
            output_dir=str(data_dir),
            timespan=CONFIG["timespan"],
            years=CONFIG["years"],
        )

    logger.info(f"Downloaded data for {len(results)} symbols")
    return results


def prepare_data():
    """Load and prepare training data."""
    logger.info("Preparing training data...")

    data_dir = Path(__file__).parent / CONFIG["data_dir"]

    # Auto-detect symbols if not specified
    symbols = CONFIG["symbols"]
    tier = CONFIG.get("tier")
    if symbols is None:
        symbols = get_available_symbols(data_dir, tier=tier)
        tier_msg = f" (tier: {tier})" if tier else ""
        logger.info(f"Auto-detected {len(symbols)} symbols from data directory{tier_msg}")

    if not symbols:
        raise ValueError("No symbols found. Run fetch_nasdaq.py first or specify symbols.")

    # Load all symbols
    all_data = []
    loaded_count = 0
    for i, symbol in enumerate(symbols):
        df = load_training_data([symbol], str(data_dir), CONFIG["timespan"])
        if df.empty:
            continue

        # Add features
        df = add_technical_indicators(df)
        df = create_target_variables(df, horizons=[CONFIG["prediction_horizon"]])

        all_data.append(df)
        loaded_count += 1

        # Progress update
        if (i + 1) % 100 == 0:
            logger.info(f"Loaded {loaded_count}/{i+1} symbols...")

    logger.info(f"Successfully loaded {loaded_count}/{len(symbols)} symbols")

    if not all_data:
        raise ValueError("No data available for training")

    # Combine all symbols
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    logger.info(f"Total data: {len(combined):,} rows from {loaded_count} symbols")

    return combined


def train_model(df: pd.DataFrame):
    """Train the prediction model."""
    logger.info("Training model...")

    feature_cols = get_feature_columns()
    target_col = f"target_return_{CONFIG['prediction_horizon']}d"

    # Split by time (train on earlier data, validate on later)
    split_idx = int(len(df) * CONFIG["train_split"])
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    # Normalize features
    train_df, val_df, norm_stats = normalize_features(train_df, val_df, feature_cols)

    # Prepare sequences
    X_train, y_train = prepare_sequences(
        train_df, feature_cols, target_col, CONFIG["sequence_length"]
    )
    X_val, y_val = prepare_sequences(
        val_df, feature_cols, target_col, CONFIG["sequence_length"]
    )

    logger.info(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Not enough data for training sequences")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=CONFIG["batch_size"],
    )

    # Create model
    input_size = len(feature_cols)
    model_kwargs = {
        "input_size": input_size,
        "hidden_size": CONFIG["hidden_size"],
        "dropout": CONFIG["dropout"],
        "output_size": 1,
    }

    # Add model-specific parameters
    if CONFIG["model_type"] == "tft":
        model_kwargs["num_heads"] = CONFIG.get("num_heads", 4)
        model_kwargs["num_encoder_layers"] = CONFIG.get("num_layers", 2)
    else:
        model_kwargs["num_layers"] = CONFIG.get("num_layers", 2)

    model = create_model(
        model_type=CONFIG["model_type"],
        **model_kwargs,
    )

    logger.info(f"Model: {CONFIG['model_type']}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    model_dir = Path(__file__).parent / CONFIG["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=CONFIG["epochs"],
        early_stopping_patience=CONFIG["early_stopping_patience"],
        checkpoint_dir=str(model_dir),
    )

    # Evaluate
    logger.info("Evaluating model...")
    y_pred = trainer.predict(X_val)
    metrics = evaluate_predictions(y_val, y_pred.squeeze())

    logger.info("Evaluation metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # Export to ONNX for C++ inference
    onnx_path = model_dir / "model.onnx"
    trainer.export_onnx(str(onnx_path), input_shape=(CONFIG["sequence_length"], input_size))

    # Save normalization stats
    import json
    with open(model_dir / "norm_stats.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in norm_stats.items()}, f)

    # Save config
    with open(model_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    logger.info(f"Model saved to {model_dir}")

    return trainer, metrics


def main():
    parser = argparse.ArgumentParser(description="Train ArgusTrader ML models")
    parser.add_argument("--download", action="store_true", help="Download training data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--symbols", nargs="+", help="Override symbols to train on")
    parser.add_argument("--model", choices=["lstm", "transformer", "cnn_lstm", "attention_lstm", "tft"],
                        help="Model type")
    parser.add_argument("--tier", choices=["blue_chip", "mid_cap", "penny_stocks", "new_listings"],
                        help="Train on specific stock tier")
    parser.add_argument("--list-tiers", action="store_true", help="List available tiers")

    args = parser.parse_args()

    if args.list_tiers:
        print("\nAvailable Stock Tiers for Training:")
        print("=" * 60)
        print("\nblue_chip:")
        print("  Large, established companies - low risk, steady patterns")
        print("  Best for: Conservative strategies, swing trading")
        print("\nmid_cap:")
        print("  Mid-sized companies - moderate risk, good liquidity")
        print("  Best for: Balanced strategies, day trading")
        print("\npenny_stocks:")
        print("  Under $5 stocks - HIGH RISK, high reward potential")
        print("  Best for: Aggressive strategies, momentum plays (like INO)")
        print("\nnew_listings:")
        print("  Recent IPOs - explosive potential, unpredictable")
        print("  Best for: Speculative plays, IPO momentum")
        print("\nUsage:")
        print("  python train.py --train --tier penny_stocks")
        print("  python train.py --train --tier new_listings")
        return

    if args.symbols:
        CONFIG["symbols"] = args.symbols
    if args.model:
        CONFIG["model_type"] = args.model
    if args.tier:
        CONFIG["tier"] = args.tier
        # Save to tier-specific model directory
        CONFIG["model_dir"] = f"models/{args.tier}"

    if not args.download and not args.train:
        parser.print_help()
        print("\nExample usage:")
        print("  python train.py --download              # Download data first")
        print("  python train.py --train                 # Train the model")
        print("  python train.py --download --train      # Both")
        return

    if args.download:
        download_data()

    if args.train:
        df = prepare_data()
        trainer, metrics = train_model(df)

        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"Model: {CONFIG['model_type']}")
        print(f"Prediction horizon: {CONFIG['prediction_horizon']} days")
        print("\nMetrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print(f"\nModel saved to: {Path(__file__).parent / CONFIG['model_dir']}")


if __name__ == "__main__":
    main()
