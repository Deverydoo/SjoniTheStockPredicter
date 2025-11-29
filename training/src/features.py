"""
Feature engineering for stock price prediction.

Generates technical indicators and price features from OHLCV data.
"""

import numpy as np
import pandas as pd
from typing import Optional


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV DataFrame.

    Expects columns: timestamp, open, high, low, close, volume
    """
    df = df.copy()

    # Price-based features
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        df[f"sma_{window}"] = df["close"].rolling(window=window).mean()
        df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()

    # Price relative to moving averages
    df["price_sma_20_ratio"] = df["close"] / df["sma_20"]
    df["price_sma_50_ratio"] = df["close"] / df["sma_50"]

    # Bollinger Bands
    df["bb_middle"] = df["sma_20"]
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # ATR (Average True Range)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(window=14).mean()

    # Volatility
    df["volatility_20"] = df["returns"].rolling(window=20).std() * np.sqrt(252)

    # Volume features
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    # Price momentum
    for days in [1, 5, 10, 20]:
        df[f"momentum_{days}"] = df["close"] / df["close"].shift(days) - 1

    # High/Low features
    df["daily_range"] = (df["high"] - df["low"]) / df["close"]
    df["high_20"] = df["high"].rolling(window=20).max()
    df["low_20"] = df["low"].rolling(window=20).min()
    df["position_in_range"] = (df["close"] - df["low_20"]) / (df["high_20"] - df["low_20"])

    # Gap features
    df["gap"] = df["open"] / df["close"].shift(1) - 1

    return df


def create_target_variables(
    df: pd.DataFrame,
    horizons: list[int] = [1, 5, 10],
    threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Create target variables for prediction.

    Args:
        df: DataFrame with OHLCV data
        horizons: Prediction horizons in days
        threshold: Threshold for classification (e.g., 0.02 = 2%)

    Returns:
        DataFrame with target columns added
    """
    df = df.copy()

    for h in horizons:
        # Regression target: future returns
        df[f"target_return_{h}d"] = df["close"].shift(-h) / df["close"] - 1

        # Classification target: up/down/flat
        future_return = df[f"target_return_{h}d"]
        df[f"target_direction_{h}d"] = 0  # flat
        df.loc[future_return > threshold, f"target_direction_{h}d"] = 1  # up
        df.loc[future_return < -threshold, f"target_direction_{h}d"] = -1  # down

        # Binary classification: up or not
        df[f"target_up_{h}d"] = (future_return > 0).astype(int)

    return df


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    sequence_length: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM/Transformer training.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        sequence_length: Number of time steps in each sequence

    Returns:
        X: (num_samples, sequence_length, num_features)
        y: (num_samples,)
    """
    # Drop rows with NaN
    data = df[feature_cols + [target_col]].dropna()

    features = data[feature_cols].values
    targets = data[target_col].values

    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(features[i : i + sequence_length])
        y.append(targets[i + sequence_length])

    return np.array(X), np.array(y)


def get_feature_columns() -> list[str]:
    """Get list of feature columns to use for training."""
    return [
        # Returns
        "returns",
        "log_returns",
        # Moving average ratios
        "price_sma_20_ratio",
        "price_sma_50_ratio",
        # Bollinger Bands
        "bb_width",
        "bb_position",
        # Momentum indicators
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        # Volatility
        "atr_14",
        "volatility_20",
        # Volume
        "volume_ratio",
        # Momentum
        "momentum_1",
        "momentum_5",
        "momentum_10",
        "momentum_20",
        # Price position
        "daily_range",
        "position_in_range",
        # Gap
        "gap",
    ]


def normalize_features(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Normalize features using z-score normalization.

    Fits on training data, transforms both train and validation.

    Returns:
        Normalized train_df, normalized val_df, normalization stats
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    # Calculate mean and std from training data only
    stats = {}
    for col in feature_cols:
        if col in train_df.columns:
            stats[col] = {
                "mean": train_df[col].mean(),
                "std": train_df[col].std(),
            }

    # Normalize training data
    train_normalized = train_df.copy()
    for col, stat in stats.items():
        if stat["std"] > 0:
            train_normalized[col] = (train_df[col] - stat["mean"]) / stat["std"]
        else:
            train_normalized[col] = 0

    # Normalize validation data using training stats
    val_normalized = None
    if val_df is not None:
        val_normalized = val_df.copy()
        for col, stat in stats.items():
            if col in val_df.columns and stat["std"] > 0:
                val_normalized[col] = (val_df[col] - stat["mean"]) / stat["std"]
            elif col in val_df.columns:
                val_normalized[col] = 0

    return train_normalized, val_normalized, stats


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    # Load sample data
    from data_fetcher import load_training_data

    df = load_training_data(
        symbols=["AAPL"],
        data_dir="d:/Vibe_Projects/The Trader/training/data",
        timespan="day",
    )

    if not df.empty:
        print(f"Loaded {len(df)} rows")

        # Add features
        df = add_technical_indicators(df)
        df = create_target_variables(df)

        print(f"\nFeatures created. Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")

        # Prepare sequences
        feature_cols = get_feature_columns()
        X, y = prepare_sequences(df, feature_cols, "target_return_5d", sequence_length=60)

        print(f"\nSequences: X={X.shape}, y={y.shape}")
