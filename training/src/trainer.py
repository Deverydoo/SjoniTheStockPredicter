"""
Training loop and utilities for stock prediction models.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training manager for PyTorch models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        logger.info(f"Using device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        criterion: Optional[nn.Module] = None,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None,
    ) -> dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            criterion: Loss function (default: MSELoss)
            early_stopping_patience: Stop if no improvement for this many epochs
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        if criterion is None:
            criterion = nn.MSELoss()

        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, criterion)

            # Validate
            val_loss = self.validate(val_loader, criterion)

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.2e}"
            )

            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                if checkpoint_dir:
                    self.save_checkpoint(
                        checkpoint_path / "best_model.pt",
                        epoch=epoch,
                        val_loss=val_loss,
                    )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return self.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()

        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            outputs = self.model(x_tensor)
            return outputs.cpu().numpy()

    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint

    def export_onnx(self, path: str, input_shape: tuple):
        """Export model to ONNX format for C++ inference."""
        self.model.eval()
        dummy_input = torch.randn(1, *input_shape).to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        logger.info(f"Exported ONNX model to {path}")


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders."""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate evaluation metrics."""
    # Regression metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # Correlation
    correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

    # Direction accuracy
    direction_correct = np.mean(np.sign(y_true) == np.sign(y_pred))

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": float(correlation),
        "direction_accuracy": float(direction_correct),
    }
