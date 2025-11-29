#!/usr/bin/env python3
"""
Sjoni Training Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

Unified training script for the 547M parameter Sjoni model.

This script:
1. Loads all historical price data from data/historical/
2. Computes price-derived sentiment features
3. Loads delisted stock data from data/delisted/
4. Builds institutional flow features
5. Creates proper PyTorch Dataset and DataLoader
6. Trains with mixed precision and gradient checkpointing
7. Saves checkpoints to models/

Usage:
    python train_sjoni.py                    # Full training
    python train_sjoni.py --test             # Quick test with small batch
    python train_sjoni.py --resume checkpoint.pt  # Resume from checkpoint

Hardware: RTX 4090 (24GB VRAM) recommended
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sjoni import Sjoni, SjoniConfig, create_sjoni_500m
from features import add_technical_indicators
from gdelt_historical import PriceDerivedSentiment
from institutional_flow import InstitutionalFlowAnalyzer
from sector_classifier import SectorClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Data
    data_dir: str = "data"
    sequence_length: int = 252  # 1 year
    prediction_horizon: int = 5
    train_split: float = 0.8
    val_split: float = 0.1  # Remaining 10% is test

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 4  # Effective batch = 128
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    # Optimization
    use_amp: bool = True  # Mixed precision
    clip_grad_norm: float = 1.0

    # Regularization
    label_smoothing: float = 0.1
    dropout: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "models"
    save_every_epochs: int = 5
    early_stopping_patience: int = 15

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True

    # Loss weights
    price_loss_weight: float = 1.0
    direction_loss_weight: float = 0.5
    regime_loss_weight: float = 0.3
    delisting_loss_weight: float = 0.5
    uncertainty_loss_weight: float = 0.2


class SjoniDataset(Dataset):
    """
    PyTorch Dataset for Sjoni training

    Loads and processes:
    - Historical price data
    - Technical indicators
    - Price-derived sentiment
    - Institutional flow features
    - Sector/industry classifications
    - Delisting risk labels (when available)
    """

    def __init__(
        self,
        data_dir: str,
        symbols: List[str],
        sequence_length: int = 252,
        prediction_horizon: int = 5,
        include_delisted: bool = True,
        mode: str = "train"  # train, val, test
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.mode = mode

        # Initialize processors
        self.sentiment_processor = PriceDerivedSentiment()
        self.institutional_analyzer = InstitutionalFlowAnalyzer()
        self.sector_classifier = SectorClassifier()

        # Build stock -> index mapping
        self.symbol_to_idx = {s: i for i, s in enumerate(symbols)}

        # Load and process all data
        logger.info(f"Loading {len(symbols)} symbols for {mode}...")
        self.samples = self._build_samples(symbols, include_delisted)
        logger.info(f"Built {len(self.samples)} training samples for {mode}")

        # Compute normalization statistics from training data
        if mode == "train":
            self.stats = self._compute_stats()
        else:
            self.stats = None

    def _load_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load price data for a symbol"""
        # Try different file patterns
        patterns = [
            self.data_dir / "historical" / f"{symbol}_day.parquet",
            self.data_dir / f"{symbol}_day.parquet",
            self.data_dir / "historical" / f"{symbol}_max.parquet",
        ]

        for path in patterns:
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    # Standardize column names
                    df.columns = [c.lower() for c in df.columns]

                    # Ensure index is datetime
                    if not isinstance(df.index, pd.DatetimeIndex):
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date')
                        elif 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df = df.set_index('timestamp')

                    return df
                except Exception as e:
                    logger.debug(f"Error loading {path}: {e}")

        return None

    def _build_samples(
        self,
        symbols: List[str],
        include_delisted: bool
    ) -> List[Dict]:
        """Build training samples from all symbols"""
        samples = []

        # Load market data for relative sentiment
        market_df = self._load_price_data("SPY")
        if market_df is None:
            market_df = self._load_price_data("QQQ")

        for symbol in symbols:
            df = self._load_price_data(symbol)
            if df is None or len(df) < self.sequence_length + self.prediction_horizon + 50:
                continue

            try:
                # Add technical indicators
                df = add_technical_indicators(df)

                # Add sentiment features
                sentiment_features = self.sentiment_processor.calculate_sentiment_proxy(df, market_df)
                df = pd.concat([df, sentiment_features], axis=1)

                # Add institutional flow features
                inst_features = self.institutional_analyzer.analyze(df)
                df = pd.concat([df, inst_features], axis=1)

                # Get sector/industry (returns sector_id, industry_id, sector_name, industry_name)
                sector_id, industry_id, _, _ = self.sector_classifier.classify(symbol)

                # Drop rows with NaN
                df = df.dropna()

                if len(df) < self.sequence_length + self.prediction_horizon:
                    continue

                # Create sliding window samples
                stock_idx = self.symbol_to_idx.get(symbol, 0)

                for i in range(len(df) - self.sequence_length - self.prediction_horizon):
                    # Get window
                    window = df.iloc[i:i + self.sequence_length]
                    future = df.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]

                    # Calculate targets
                    current_price = window['close'].iloc[-1]
                    future_prices = future['close'].values
                    future_returns = (future_prices / current_price) - 1

                    samples.append({
                        'symbol': symbol,
                        'stock_idx': stock_idx,
                        'sector_id': sector_id,
                        'industry_id': industry_id,
                        'date': window.index[-1],
                        'window': window,
                        'target_returns': future_returns,
                        'target_direction': (future_returns > 0).astype(float),
                        'is_delisted': False
                    })

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        # Load delisted stocks
        if include_delisted:
            delisted_samples = self._load_delisted_samples()
            samples.extend(delisted_samples)

        return samples

    def _load_delisted_samples(self) -> List[Dict]:
        """Load samples from delisted stocks"""
        samples = []

        delisted_dir = self.data_dir / "delisted"
        prices_file = delisted_dir / "delisted_prices.parquet"
        labels_file = delisted_dir / "delisting_labels.parquet"

        if not prices_file.exists():
            logger.info("No delisted data found, skipping")
            return samples

        try:
            prices_df = pd.read_parquet(prices_file)
            labels_df = pd.read_parquet(labels_file) if labels_file.exists() else None

            for symbol in prices_df['symbol'].unique():
                stock_data = prices_df[prices_df['symbol'] == symbol].copy()

                if len(stock_data) < self.sequence_length + self.prediction_horizon:
                    continue

                # Process like regular stocks
                stock_data = add_technical_indicators(stock_data)
                sentiment = self.sentiment_processor.calculate_sentiment_proxy(stock_data)
                stock_data = pd.concat([stock_data, sentiment], axis=1)
                inst_features = self.institutional_analyzer.analyze(stock_data)
                stock_data = pd.concat([stock_data, inst_features], axis=1)
                stock_data = stock_data.dropna()

                if len(stock_data) < self.sequence_length + self.prediction_horizon:
                    continue

                # Create samples with delisting labels
                for i in range(len(stock_data) - self.sequence_length - self.prediction_horizon):
                    window = stock_data.iloc[i:i + self.sequence_length]
                    future = stock_data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]

                    current_price = window['close'].iloc[-1]
                    future_returns = (future['close'].values / current_price) - 1

                    # Get delisting labels if available
                    delist_risk = np.zeros(4)  # 30d, 90d, 180d, 365d
                    if labels_df is not None:
                        date = window.index[-1]
                        label_row = labels_df[
                            (labels_df['symbol'] == symbol) &
                            (labels_df['date'] == date)
                        ]
                        if not label_row.empty:
                            for j, col in enumerate(['delist_30d', 'delist_90d', 'delist_180d', 'delist_365d']):
                                if col in label_row.columns:
                                    delist_risk[j] = label_row[col].values[0]

                    samples.append({
                        'symbol': symbol,
                        'stock_idx': len(self.symbol_to_idx),  # Out-of-vocabulary
                        'sector_id': 0,  # Unknown
                        'industry_id': 0,
                        'date': window.index[-1],
                        'window': window,
                        'target_returns': future_returns,
                        'target_direction': (future_returns > 0).astype(float),
                        'delist_risk': delist_risk,
                        'is_delisted': True
                    })

            logger.info(f"Loaded {len(samples)} delisted stock samples")

        except Exception as e:
            logger.warning(f"Error loading delisted data: {e}")

        return samples

    def _compute_stats(self) -> Dict:
        """Compute normalization statistics"""
        # Collect all feature values
        price_features = []
        sentiment_features = []
        institutional_features = []

        for sample in self.samples[:min(len(self.samples), 10000)]:  # Sample subset
            window = sample['window']

            price_cols = self._get_price_columns(window)
            sent_cols = self._get_sentiment_columns(window)
            inst_cols = self._get_institutional_columns(window)

            if price_cols:
                price_features.append(window[price_cols].values)
            if sent_cols:
                sentiment_features.append(window[sent_cols].values)
            if inst_cols:
                institutional_features.append(window[inst_cols].values)

        stats = {}

        if price_features:
            price_all = np.concatenate(price_features, axis=0)
            stats['price_mean'] = np.nanmean(price_all, axis=0)
            stats['price_std'] = np.nanstd(price_all, axis=0) + 1e-8

        if sentiment_features:
            sent_all = np.concatenate(sentiment_features, axis=0)
            stats['sentiment_mean'] = np.nanmean(sent_all, axis=0)
            stats['sentiment_std'] = np.nanstd(sent_all, axis=0) + 1e-8

        if institutional_features:
            inst_all = np.concatenate(institutional_features, axis=0)
            stats['institutional_mean'] = np.nanmean(inst_all, axis=0)
            stats['institutional_std'] = np.nanstd(inst_all, axis=0) + 1e-8

        return stats

    def set_stats(self, stats: Dict):
        """Set normalization statistics (for val/test sets)"""
        self.stats = stats

    def _get_price_columns(self, df: pd.DataFrame) -> List[str]:
        """Get price-related feature columns"""
        price_cols = [
            'returns', 'log_returns', 'price_sma_20_ratio', 'price_sma_50_ratio',
            'bb_width', 'bb_position', 'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'atr_14', 'volatility_20', 'volume_ratio', 'momentum_1', 'momentum_5',
            'momentum_10', 'momentum_20', 'daily_range', 'position_in_range', 'gap'
        ]
        return [c for c in price_cols if c in df.columns]

    def _get_sentiment_columns(self, df: pd.DataFrame) -> List[str]:
        """Get sentiment feature columns"""
        sent_cols = [
            'overnight_gap', 'gap_direction', 'close_position', 'bullish_volume',
            'bearish_volume', 'fear_signal', 'sentiment_proxy', 'sentiment_proxy_smooth'
        ]
        return [c for c in sent_cols if c in df.columns]

    def _get_institutional_columns(self, df: pd.DataFrame) -> List[str]:
        """Get institutional flow feature columns"""
        inst_cols = [
            'unusual_volume', 'block_trade_prob', 'accumulation_distribution',
            'money_flow_index', 'smart_money_index', 'vwap_divergence',
            'obv_trend', 'pv_divergence', 'weak_hands_score', 'squeeze_potential',
            'smart_money_flow', 'retail_flow'
        ]
        return [c for c in inst_cols if c in df.columns]

    def _get_market_columns(self, df: pd.DataFrame) -> List[str]:
        """Get market context feature columns"""
        # For now, we derive market features from the stock data
        # In production, these would come from VIX, sector ETFs, etc.
        return ['volatility_20', 'volume_ratio']

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        window = sample['window']

        # Extract features
        price_cols = self._get_price_columns(window)
        sent_cols = self._get_sentiment_columns(window)
        inst_cols = self._get_institutional_columns(window)

        # Get feature arrays
        price_features = window[price_cols].values if price_cols else np.zeros((len(window), 32))
        sent_features = window[sent_cols].values if sent_cols else np.zeros((len(window), 8))
        inst_features = window[inst_cols].values if inst_cols else np.zeros((len(window), 12))

        # Pad to expected dimensions
        price_features = self._pad_features(price_features, 32)
        sent_features = self._pad_features(sent_features, 8)
        inst_features = self._pad_features(inst_features, 12)
        market_features = self._pad_features(
            window[self._get_market_columns(window)].values if self._get_market_columns(window) else np.zeros((len(window), 2)),
            16
        )

        # Normalize using training stats
        if self.stats:
            if 'price_mean' in self.stats:
                price_features = (price_features - self.stats['price_mean'][:price_features.shape[1]]) / self.stats['price_std'][:price_features.shape[1]]
            if 'sentiment_mean' in self.stats:
                sent_features = (sent_features - self.stats['sentiment_mean'][:sent_features.shape[1]]) / self.stats['sentiment_std'][:sent_features.shape[1]]
            if 'institutional_mean' in self.stats:
                inst_features = (inst_features - self.stats['institutional_mean'][:inst_features.shape[1]]) / self.stats['institutional_std'][:inst_features.shape[1]]

        # Handle NaN/Inf
        price_features = np.nan_to_num(price_features, nan=0.0, posinf=3.0, neginf=-3.0)
        sent_features = np.nan_to_num(sent_features, nan=0.0, posinf=3.0, neginf=-3.0)
        inst_features = np.nan_to_num(inst_features, nan=0.0, posinf=3.0, neginf=-3.0)
        market_features = np.nan_to_num(market_features, nan=0.0, posinf=3.0, neginf=-3.0)

        # Calculate stock age (months since first available data)
        stock_age = min(99, len(window) // 21)  # Approximate months

        return {
            'price_features': torch.FloatTensor(price_features),
            'sentiment_features': torch.FloatTensor(sent_features),
            'institutional_features': torch.FloatTensor(inst_features),
            'market_features': torch.FloatTensor(market_features),
            'sector_ids': torch.LongTensor([sample['sector_id']]),
            'industry_ids': torch.LongTensor([sample['industry_id']]),
            'stock_age_months': torch.LongTensor([stock_age]),
            'stock_ids': torch.LongTensor([sample['stock_idx']]),
            'target_returns': torch.FloatTensor(sample['target_returns']),
            'target_direction': torch.FloatTensor(sample['target_direction']),
            'delist_risk': torch.FloatTensor(sample.get('delist_risk', np.zeros(4))),
            'is_delisted': torch.BoolTensor([sample.get('is_delisted', False)])
        }

    def _pad_features(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad features to target dimension"""
        current_dim = features.shape[1] if features.ndim > 1 else 1
        if current_dim >= target_dim:
            return features[:, :target_dim]
        else:
            padding = np.zeros((features.shape[0], target_dim - current_dim))
            return np.concatenate([features, padding], axis=1)


class SjoniLoss(nn.Module):
    """
    Combined loss function for Sjoni

    Components:
    1. Price prediction loss (MSE + direction accuracy)
    2. Uncertainty calibration loss (NLL)
    3. Regime classification loss (CrossEntropy)
    4. Delisting risk loss (BCE)
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # Loss components
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss"""
        losses = {}

        # 1. Price prediction loss
        price_pred = outputs['price_prediction']
        price_target = targets['target_returns']

        # MSE loss on returns
        price_mse = self.mse_loss(price_pred, price_target)
        losses['price_mse'] = price_mse.item()

        # Direction accuracy as auxiliary loss
        pred_direction = (price_pred > 0).float()
        target_direction = targets['target_direction']
        direction_loss = self.bce_loss(price_pred, target_direction)
        losses['direction_bce'] = direction_loss.item()

        # 2. Uncertainty calibration (Gaussian NLL)
        if 'price_uncertainty' in outputs:
            uncertainty = outputs['price_uncertainty']
            # NLL for Gaussian: log(sigma) + (y - mu)^2 / (2 * sigma^2)
            nll = torch.log(uncertainty + 1e-8) + (price_pred - price_target) ** 2 / (2 * uncertainty ** 2 + 1e-8)
            uncertainty_loss = nll.mean()
            losses['uncertainty_nll'] = uncertainty_loss.item()
        else:
            uncertainty_loss = torch.tensor(0.0, device=price_pred.device)

        # 3. Delisting risk loss
        if 'delist_risk' in targets and 'delisting_risk' in outputs:
            delist_pred = outputs['delisting_risk']
            delist_target = targets['delist_risk']

            # Weight by is_delisted flag for balanced training
            is_delisted = targets.get('is_delisted', torch.zeros_like(delist_target[:, 0]))

            # BCE loss with higher weight for delisted samples
            weight = torch.where(is_delisted.bool().squeeze(),
                                torch.tensor(5.0, device=delist_pred.device),
                                torch.tensor(1.0, device=delist_pred.device))

            delist_loss = F.binary_cross_entropy(
                delist_pred, delist_target,
                weight=weight.unsqueeze(-1).expand_as(delist_pred),
                reduction='mean'
            )
            losses['delist_bce'] = delist_loss.item()
        else:
            delist_loss = torch.tensor(0.0, device=price_pred.device)

        # Combine losses
        total_loss = (
            self.config.price_loss_weight * price_mse +
            self.config.direction_loss_weight * direction_loss +
            self.config.uncertainty_loss_weight * uncertainty_loss +
            self.config.delisting_loss_weight * delist_loss
        )

        losses['total'] = total_loss.item()

        return total_loss, losses


class SjoniTrainer:
    """
    Training manager for Sjoni

    Features:
    - Mixed precision training (AMP)
    - Gradient checkpointing
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpoint management
    - Metrics tracking
    """

    def __init__(
        self,
        model: Sjoni,
        config: TrainingConfig,
        device: torch.device
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Loss function
        self.criterion = SjoniLoss(config)

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Scheduler (will be set after dataloader is known)
        self.scheduler = None

        # Metrics tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'metrics': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def setup_scheduler(self, steps_per_epoch: int):
        """Setup learning rate scheduler"""
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1000
        )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        epoch_losses = []
        accumulation_counter = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Squeeze single dimensions
            for key in ['sector_ids', 'industry_ids', 'stock_age_months', 'stock_ids']:
                if key in batch:
                    batch[key] = batch[key].squeeze(-1)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(
                    price_features=batch['price_features'],
                    sentiment_features=batch['sentiment_features'],
                    institutional_features=batch['institutional_features'],
                    market_features=batch['market_features'],
                    sector_ids=batch['sector_ids'],
                    industry_ids=batch['industry_ids'],
                    stock_age_months=batch['stock_age_months'],
                    stock_ids=batch['stock_ids']
                )

                loss, loss_dict = self.criterion(outputs, batch)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_counter += 1

            # Optimizer step after accumulation
            if accumulation_counter >= self.config.gradient_accumulation_steps:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.clip_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.clip_grad_norm
                    )
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                accumulation_counter = 0

            epoch_losses.append(loss_dict)

            # Progress logging
            if batch_idx % 100 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"  Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss_dict['total']:.4f} | "
                    f"LR: {lr:.2e}"
                )

        # Average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])

        return avg_losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        val_losses = []
        all_preds = []
        all_targets = []

        for batch in val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            for key in ['sector_ids', 'industry_ids', 'stock_age_months', 'stock_ids']:
                if key in batch:
                    batch[key] = batch[key].squeeze(-1)

            with autocast(enabled=self.config.use_amp):
                outputs = self.model(
                    price_features=batch['price_features'],
                    sentiment_features=batch['sentiment_features'],
                    institutional_features=batch['institutional_features'],
                    market_features=batch['market_features'],
                    sector_ids=batch['sector_ids'],
                    industry_ids=batch['industry_ids'],
                    stock_age_months=batch['stock_age_months'],
                    stock_ids=batch['stock_ids']
                )

                loss, loss_dict = self.criterion(outputs, batch)

            val_losses.append(loss_dict)

            # Collect predictions for metrics
            all_preds.append(outputs['price_prediction'].cpu())
            all_targets.append(batch['target_returns'].cpu())

        # Average losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in val_losses])

        # Calculate additional metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Direction accuracy
        pred_direction = (all_preds > 0).float()
        target_direction = (all_targets > 0).float()
        direction_acc = (pred_direction == target_direction).float().mean().item()
        avg_losses['direction_accuracy'] = direction_acc

        # Correlation
        for i in range(min(5, all_preds.shape[1])):
            corr = np.corrcoef(
                all_preds[:, i].numpy().flatten(),
                all_targets[:, i].numpy().flatten()
            )[0, 1]
            avg_losses[f'correlation_day{i+1}'] = corr if not np.isnan(corr) else 0.0

        return avg_losses

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'val_loss': val_loss,
            'config': self.config.__dict__,
            'history': self.history
        }

        # Save latest
        latest_path = Path(self.config.checkpoint_dir) / "sjoni_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save epoch checkpoint
        if epoch % self.config.save_every_epochs == 0:
            epoch_path = Path(self.config.checkpoint_dir) / f"sjoni_epoch_{epoch}.pt"
            torch.save(checkpoint, epoch_path)

        # Save best
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "sjoni_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.history = checkpoint.get('history', self.history)

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        return checkpoint['epoch']

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 0
    ):
        """Main training loop"""
        # Setup scheduler
        self.setup_scheduler(len(train_loader))

        logger.info("=" * 60)
        logger.info("Starting Sjoni Training")
        logger.info("=" * 60)
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 60)

        for epoch in range(start_epoch, self.config.epochs):
            epoch_start = datetime.now()

            logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            logger.info("-" * 40)

            # Train
            train_losses = self.train_epoch(train_loader)

            # Validate
            val_losses = self.validate(val_loader)

            # Record history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['metrics'].append(val_losses)

            # Log epoch summary
            epoch_time = datetime.now() - epoch_start
            logger.info(f"\nEpoch {epoch + 1} Summary:")
            logger.info(f"  Train Loss: {train_losses['total']:.4f}")
            logger.info(f"  Val Loss:   {val_losses['total']:.4f}")
            logger.info(f"  Direction Acc: {val_losses.get('direction_accuracy', 0):.2%}")
            logger.info(f"  Day 1 Corr: {val_losses.get('correlation_day1', 0):.3f}")
            logger.info(f"  Time: {epoch_time}")

            # Check for improvement
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch + 1, val_losses['total'], is_best)

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Best Val Loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)


def get_symbols(data_dir: str) -> List[str]:
    """Get list of available symbols from data directory"""
    symbols = set()

    # Check historical directory
    historical_dir = Path(data_dir) / "historical"
    if historical_dir.exists():
        for f in historical_dir.glob("*_day.parquet"):
            symbol = f.stem.replace("_day", "")
            symbols.add(symbol)

    # Check root data directory
    for f in Path(data_dir).glob("*_day.parquet"):
        symbol = f.stem.replace("_day", "")
        symbols.add(symbol)

    return sorted(list(symbols))


def split_symbols(
    symbols: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Split symbols into train/val/test sets"""
    np.random.seed(seed)
    np.random.shuffle(symbols)

    n = len(symbols)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_symbols = symbols[:n_train]
    val_symbols = symbols[n_train:n_train + n_val]
    test_symbols = symbols[n_train + n_val:]

    return train_symbols, val_symbols, test_symbols


def main():
    parser = argparse.ArgumentParser(description="Train Sjoni 547M Model")
    parser.add_argument('--test', action='store_true', help='Quick test run')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    args = parser.parse_args()

    # Configuration
    config = TrainingConfig()
    config.data_dir = args.data_dir

    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.no_amp:
        config.use_amp = False

    if args.test:
        config.epochs = 2
        config.batch_size = 8
        config.gradient_accumulation_steps = 1

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Get symbols
    logger.info(f"Loading symbols from {config.data_dir}...")
    symbols = get_symbols(config.data_dir)

    if args.test:
        symbols = symbols[:50]  # Small subset for testing

    logger.info(f"Found {len(symbols)} symbols")

    if len(symbols) == 0:
        logger.error("No data found! Please run fetch_historical.py first.")
        return

    # Split symbols
    train_symbols, val_symbols, test_symbols = split_symbols(
        symbols, config.train_split, config.val_split
    )

    logger.info(f"Train: {len(train_symbols)}, Val: {len(val_symbols)}, Test: {len(test_symbols)}")

    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset = SjoniDataset(
        data_dir=config.data_dir,
        symbols=train_symbols,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
        include_delisted=True,
        mode="train"
    )

    logger.info("Creating validation dataset...")
    val_dataset = SjoniDataset(
        data_dir=config.data_dir,
        symbols=val_symbols,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
        include_delisted=False,
        mode="val"
    )
    val_dataset.set_stats(train_dataset.stats)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # Create model
    logger.info("Creating Sjoni 547M model...")
    model = create_sjoni_500m()

    # Create trainer
    trainer = SjoniTrainer(model, config, device)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    # Train!
    trainer.fit(train_loader, val_loader, start_epoch)

    # Save final model info
    info = {
        'final_val_loss': trainer.best_val_loss,
        'epochs_trained': len(trainer.history['train_loss']),
        'train_symbols': len(train_symbols),
        'val_symbols': len(val_symbols),
        'total_samples': len(train_dataset) + len(val_dataset),
        'config': config.__dict__
    }

    with open(Path(config.checkpoint_dir) / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2, default=str)

    logger.info("\nTraining artifacts saved to: models/")
    logger.info("  - sjoni_best.pt (best model)")
    logger.info("  - sjoni_latest.pt (latest checkpoint)")
    logger.info("  - training_info.json (training metadata)")


if __name__ == "__main__":
    main()
