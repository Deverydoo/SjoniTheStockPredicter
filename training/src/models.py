"""
PyTorch models for stock price prediction.

Includes:
- LSTM: Long Short-Term Memory network
- Transformer: Attention-based model
- TFT: Temporal Fusion Transformer (state-of-the-art)
- Ensemble: Combines multiple models
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LSTMPredictor(nn.Module):
    """
    LSTM-based price prediction model.

    Architecture:
    - Multi-layer LSTM
    - Dropout for regularization
    - Fully connected output layer
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        # Output layer
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            (batch_size, output_size)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward last hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # Apply dropout and FC
        out = self.dropout(hidden)
        out = self.fc(out)

        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer-based price prediction model.

    Uses self-attention to capture temporal dependencies.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_size: int = 1,
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            (batch_size, output_size)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Use last position for prediction
        x = x[:, -1, :]

        # Output
        return self.fc(x)


class CNNLSTMPredictor(nn.Module):
    """
    Hybrid CNN-LSTM model.

    Uses CNN for local pattern extraction, LSTM for temporal modeling.
    """

    def __init__(
        self,
        input_size: int,
        cnn_channels: list[int] = [32, 64],
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()

        # CNN layers
        cnn_layers = []
        in_channels = input_size
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Output
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            (batch_size, output_size)
        """
        # CNN expects (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        # Back to (batch, length, channels)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]

        # Output
        return self.fc(hidden)


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism.

    Allows the model to focus on the most relevant time steps.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            (batch_size, output_size)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)

        # Output
        return self.fc(context)


# =============================================================================
# Temporal Fusion Transformer Components
# =============================================================================

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for TFT."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.sigmoid(self.fc1(x)) * self.fc2(x))


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - key component of TFT.

    Provides flexible nonlinear processing with skip connections
    and gating for adaptive depth.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = None,
        context_size: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.context_size = context_size
        self.hidden_size = hidden_size

        # Main layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)

        # Context projection (if context is provided)
        if context_size is not None:
            self.context_projection = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_projection = None

        # Gate
        self.gate = GatedLinearUnit(self.output_size, self.output_size, dropout)

        # Skip connection (if input/output sizes differ)
        if input_size != self.output_size:
            self.skip = nn.Linear(input_size, self.output_size)
        else:
            self.skip = None

        # Layer norm
        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None
    ) -> torch.Tensor:
        # Skip connection
        if self.skip is not None:
            residual = self.skip(x)
        else:
            residual = x

        # Main path
        hidden = self.fc1(x)
        if self.context_projection is not None and context is not None:
            hidden = hidden + self.context_projection(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        # Gating
        gated = self.gate(hidden)

        # Add & Norm
        return self.layer_norm(gated + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) - learns which variables are most important.

    Provides interpretable feature importance weights.
    """

    def __init__(
        self,
        input_size: int,
        num_inputs: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.input_size = input_size

        # GRN for each input variable
        self.grn_vars = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
            for _ in range(num_inputs)
        ])

        # GRN for variable weights
        self.grn_weights = GatedResidualNetwork(
            input_size=num_inputs * input_size,
            hidden_size=hidden_size,
            output_size=num_inputs,
            context_size=context_size,
            dropout=dropout,
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, num_inputs, input_size) or (batch, num_inputs, input_size)
            context: optional context tensor

        Returns:
            selected: weighted combination of inputs
            weights: variable importance weights
        """
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            # (batch, num_inputs, input_size)
            batch_size = x.size(0)

            # Flatten for weight computation
            flat_x = x.reshape(batch_size, -1)

            # Compute variable weights
            weights = self.grn_weights(flat_x, context)
            weights = self.softmax(weights)  # (batch, num_inputs)

            # Process each variable
            processed = []
            for i, grn in enumerate(self.grn_vars):
                processed.append(grn(x[:, i, :]))  # (batch, hidden)
            processed = torch.stack(processed, dim=1)  # (batch, num_inputs, hidden)

            # Weighted sum
            weights_expanded = weights.unsqueeze(-1)  # (batch, num_inputs, 1)
            selected = (processed * weights_expanded).sum(dim=1)  # (batch, hidden)

        else:
            # (batch, time, num_inputs, input_size)
            batch_size, time_steps = x.size(0), x.size(1)

            # Process time-distributed
            flat_x = x.reshape(batch_size * time_steps, self.num_inputs, self.input_size)
            flat_x_concat = flat_x.reshape(batch_size * time_steps, -1)

            # Compute weights
            if context is not None:
                context_expanded = context.unsqueeze(1).expand(-1, time_steps, -1)
                context_flat = context_expanded.reshape(batch_size * time_steps, -1)
            else:
                context_flat = None

            weights = self.grn_weights(flat_x_concat, context_flat)
            weights = self.softmax(weights)

            # Process each variable
            processed = []
            for i, grn in enumerate(self.grn_vars):
                var_input = flat_x[:, i, :]
                processed.append(grn(var_input))
            processed = torch.stack(processed, dim=1)

            # Weighted sum
            weights_expanded = weights.unsqueeze(-1)
            selected = (processed * weights_expanded).sum(dim=1)
            selected = selected.reshape(batch_size, time_steps, -1)
            weights = weights.reshape(batch_size, time_steps, -1)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention for TFT.

    Uses additive attention instead of scaled dot-product
    for better interpretability of attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.size(0), query.size(1)

        # Project and reshape to (batch, heads, seq, head_dim)
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask (causal)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out_proj(context)

        # Return mean attention across heads for interpretability
        mean_attn = attn_weights.mean(dim=1)

        return output, mean_attn


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) - State-of-the-art time series model.

    Key features:
    - Variable selection networks for interpretable feature importance
    - Gated residual networks for efficient learning
    - Multi-head attention for temporal patterns
    - Quantile outputs for prediction intervals (optional)

    Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon
    Time Series Forecasting" (Lim et al., 2019)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
        num_static_features: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.output_size = output_size

        # Input embedding (project each feature to hidden_size)
        self.input_embedding = nn.Linear(1, hidden_size)

        # Variable Selection Network for temporal features
        self.vsn_temporal = VariableSelectionNetwork(
            input_size=hidden_size,
            num_inputs=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # LSTM encoder for local patterns
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )

        # Gated skip connection after LSTM
        self.lstm_gate = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.lstm_norm = nn.LayerNorm(hidden_size)

        # Static enrichment GRN
        self.static_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # Temporal self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': InterpretableMultiHeadAttention(
                    d_model=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                ),
                'gate': GatedLinearUnit(hidden_size, hidden_size, dropout),
                'norm': nn.LayerNorm(hidden_size),
                'ff': GatedResidualNetwork(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 4,
                    output_size=hidden_size,
                    dropout=dropout,
                ),
            })
            for _ in range(num_encoder_layers)
        ])

        # Output layers
        self.output_gate = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_size)

        # For storing attention weights (interpretability)
        self.attention_weights = None
        self.variable_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            (batch_size, output_size)
        """
        batch_size, seq_len, num_features = x.shape

        # Embed each feature separately
        # x: (batch, seq, features) -> (batch, seq, features, hidden)
        x_embedded = self.input_embedding(x.unsqueeze(-1))

        # Variable selection (learn feature importance)
        # (batch, seq, features, hidden) -> (batch, seq, hidden)
        x_selected, var_weights = self.vsn_temporal(x_embedded)
        self.variable_weights = var_weights  # Store for interpretability

        # LSTM for local temporal patterns
        lstm_out, _ = self.lstm_encoder(x_selected)

        # Gated skip connection
        lstm_gated = self.lstm_gate(lstm_out)
        lstm_out = self.lstm_norm(lstm_gated + x_selected)

        # Static enrichment (even without static features, adds nonlinearity)
        enriched = self.static_enrichment(lstm_out)

        # Create causal mask for self-attention
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        causal_mask = ~causal_mask  # Invert: True where attention is allowed

        # Temporal self-attention layers
        temporal_out = enriched
        all_attention_weights = []

        for layer in self.attention_layers:
            # Self-attention
            attn_out, attn_weights = layer['attention'](
                temporal_out, temporal_out, temporal_out,
                mask=causal_mask
            )
            all_attention_weights.append(attn_weights)

            # Gated skip connection
            gated = layer['gate'](attn_out)
            temporal_out = layer['norm'](gated + temporal_out)

            # Position-wise feed-forward
            temporal_out = layer['ff'](temporal_out)

        # Store attention weights for interpretability
        self.attention_weights = torch.stack(all_attention_weights, dim=1)

        # Take the last time step for prediction
        final_hidden = temporal_out[:, -1, :]

        # Output projection with gating
        output = self.output_gate(final_hidden)
        output = self.output_norm(output + final_hidden)
        output = self.output_fc(output)

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return attention weights for interpretability."""
        return self.attention_weights

    def get_variable_importance(self) -> Optional[torch.Tensor]:
        """Return variable selection weights for feature importance."""
        return self.variable_weights


def create_model(
    model_type: str,
    input_size: int,
    output_size: int = 1,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: "lstm", "transformer", "cnn_lstm", or "attention_lstm"
        input_size: Number of input features
        output_size: Number of outputs (1 for regression, 3 for classification)
        **kwargs: Model-specific arguments
    """
    models = {
        "lstm": LSTMPredictor,
        "transformer": TransformerPredictor,
        "cnn_lstm": CNNLSTMPredictor,
        "attention_lstm": AttentionLSTM,
        "tft": TemporalFusionTransformer,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](input_size=input_size, output_size=output_size, **kwargs)


if __name__ == "__main__":
    # Test models
    batch_size = 32
    seq_length = 60
    input_size = 21  # Number of features

    x = torch.randn(batch_size, seq_length, input_size)

    print("Testing models...")

    for model_type in ["lstm", "transformer", "cnn_lstm", "attention_lstm", "tft"]:
        model = create_model(model_type, input_size=input_size)
        out = model(x)
        print(f"{model_type}: input={x.shape} -> output={out.shape}")

    print("\nAll models working!")
