"""
Sjoni - The Market Seer
~~~~~~~~~~~~~~~~~~~~~~~
(Old Norse: Sjóni = "The Seer")

A 547M parameter Temporal Fusion Transformer for market intelligence.

Sjoni sees what others miss:
- 25 years of price/volume patterns (1999-present)
- News sentiment via local Ollama LLM
- Cross-stock relationships (learned, not hardcoded)
- Institutional flow detection (smart money vs weak hands)
- Market regime shifts (bubble/crash/recovery/normal)
- Delisting risk assessment (survivorship bias eliminated)

Hardware: NVIDIA RTX 4090 (24GB VRAM)
Training: 5,219 NASDAQ symbols + 500 synthetic delistings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SjoniConfig:
    """Configuration for Sjoni 547M model"""

    # Core dimensions - SCALED UP for 500M parameters
    hidden_size: int = 1280             # Main transformer hidden size (was 768)
    num_layers: int = 24                # Transformer layers (was 16)
    num_heads: int = 16                 # Attention heads (was 12)
    intermediate_size: int = 5120       # FFN intermediate size (4x hidden)
    dropout: float = 0.1

    # Sequence settings
    max_seq_length: int = 252           # ~1 year of trading days
    prediction_horizon: int = 5         # Predict 5 days ahead

    # Input features
    num_price_features: int = 32        # OHLCV + technical indicators
    num_sentiment_features: int = 20    # Sentiment features (17 from PriceDerivedSentiment + padding)
    num_institutional_features: int = 12 # Volume patterns, options flow
    num_market_features: int = 16       # VIX, sector indices, breadth

    # Embeddings - SCALED UP
    num_sectors: int = 11               # GICS sectors
    num_industries: int = 69            # GICS industries
    sector_embed_dim: int = 128         # Was 64
    industry_embed_dim: int = 256       # Was 128

    # Cross-stock attention - SCALED UP
    max_related_stocks: int = 200       # Was 50 - more related stocks for cross-correlation
    cross_stock_heads: int = 32         # Was 8 - more attention heads for relationships
    num_stocks: int = 6000              # Total stocks in universe (for learned embeddings)
    stock_embed_dim: int = 256          # Stock identity embedding dimension
    use_learned_relationships: bool = True  # Use learned vs explicit relationships

    # Output heads
    num_regime_classes: int = 6         # Bubble, Crash, Recovery, Normal, IPO-spike, IPO-decline

    # Memory optimization flags
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True    # If available

    # Approximate parameter count target: 500M

    def __post_init__(self):
        self.total_input_dim = (
            self.num_price_features +
            self.num_sentiment_features +
            self.num_institutional_features +
            self.num_market_features +
            self.sector_embed_dim +
            self.industry_embed_dim
        )
        self.head_dim = self.hidden_size // self.num_heads


class RotaryPositionalEmbedding(nn.Module):
    """RoPE - Better positional encoding for long sequences"""

    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GatedResidualNetwork(nn.Module):
    """GRN from Temporal Fusion Transformer - enhanced version"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)

        self.gate = nn.Linear(hidden_size, output_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        if input_size != output_size:
            self.skip = nn.Linear(input_size, output_size)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = self.skip(x) if self.skip else x

        hidden = F.elu(self.fc1(x))

        if context is not None and self.context_size:
            hidden = hidden + self.context_fc(context)

        hidden = F.elu(self.fc2(hidden))
        hidden = self.dropout(hidden)

        gate_out = self.gate(hidden)
        gate, hidden = gate_out.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        output = gate * hidden + (1 - gate) * residual
        return self.layer_norm(output)


class VariableSelectionNetwork(nn.Module):
    """VSN - Learns which features matter most"""

    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int = None
    ):
        super().__init__()

        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.num_inputs = len(input_sizes)

        # Transform each input to hidden size
        self.input_grns = nn.ModuleDict({
            name: GatedResidualNetwork(size, hidden_size, hidden_size, dropout, context_size)
            for name, size in input_sizes.items()
        })

        # Softmax weights for variable selection
        self.flattened_size = self.num_inputs * hidden_size
        self.selection_grn = GatedResidualNetwork(
            self.flattened_size, hidden_size, self.num_inputs, dropout, context_size
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Transform each input
        transformed = []
        for name in self.input_sizes.keys():
            transformed.append(self.input_grns[name](inputs[name], context))

        # Stack: (batch, seq, num_inputs, hidden)
        stacked = torch.stack(transformed, dim=-2)

        # Flatten for selection weights
        batch_size = stacked.shape[0]
        seq_len = stacked.shape[1] if stacked.dim() == 4 else 1

        if stacked.dim() == 4:
            flattened = stacked.view(batch_size, seq_len, -1)
        else:
            flattened = stacked.view(batch_size, -1)

        # Get selection weights
        weights = self.selection_grn(flattened, context)
        weights = F.softmax(weights, dim=-1)

        # Apply weights
        if stacked.dim() == 4:
            weights = weights.unsqueeze(-1)  # (batch, seq, num_inputs, 1)
            output = (stacked * weights).sum(dim=-2)  # (batch, seq, hidden)
        else:
            weights = weights.unsqueeze(-1)
            output = (stacked * weights).sum(dim=-2)

        return output, weights.squeeze(-1)


class LearnedStockEmbeddings(nn.Module):
    """
    Learned embeddings for all stocks in the universe.

    Instead of requiring pre-computed related stock features, the model learns:
    1. A unique embedding for each stock that captures its "identity"
    2. These embeddings naturally cluster by sector, behavior, and correlation
    3. The model can discover relationships during training via attention

    This allows the model to self-identify which stocks are related without
    explicit relationship labels.
    """

    def __init__(
        self,
        num_stocks: int,
        embed_dim: int,
        hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_stocks = num_stocks
        self.embed_dim = embed_dim

        # Core stock identity embedding
        self.stock_embed = nn.Embedding(num_stocks, embed_dim)

        # Project to hidden size for fusion with temporal features
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        # Initialize embeddings with small variance
        nn.init.normal_(self.stock_embed.weight, mean=0, std=0.02)

    def forward(self, stock_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for given stock IDs

        Args:
            stock_ids: (batch,) tensor of stock indices

        Returns:
            embeddings: (batch, hidden_size) stock identity embeddings
        """
        embed = self.stock_embed(stock_ids)  # (batch, embed_dim)
        return self.projection(embed)  # (batch, hidden_size)

    def get_all_embeddings(self) -> torch.Tensor:
        """Get embeddings for all stocks (for relationship discovery)"""
        all_ids = torch.arange(self.num_stocks, device=self.stock_embed.weight.device)
        return self.projection(self.stock_embed(all_ids))  # (num_stocks, hidden_size)


class LearnedCrossStockAttention(nn.Module):
    """
    Cross-stock attention with learned relationships.

    Key insight: Instead of providing pre-computed related stocks, we let the model
    learn which stocks are related by attending over ALL stock embeddings.

    The attention mechanism naturally learns:
    - Same-sector relationships (stocks that move together)
    - Supply chain relationships (suppliers/customers)
    - Competitor relationships (inverse correlation during specific events)
    - Anchor stock influence (AAPL, MSFT, etc. affecting the broader market)

    During training, the model discovers these relationships from the data.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 32,
        num_stocks: int = 6000,
        stock_embed_dim: int = 256,
        top_k_attend: int = 200,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.top_k = top_k_attend
        self.num_stocks = num_stocks

        # Learned stock embeddings
        self.stock_embeddings = LearnedStockEmbeddings(
            num_stocks=num_stocks,
            embed_dim=stock_embed_dim,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # Query projection (from current stock's temporal features)
        self.q_proj = nn.Linear(hidden_size, hidden_size)

        # Key/Value projections (from stock embeddings)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Learnable "relationship type" modulation
        # The model learns to weight different relationship patterns
        self.relationship_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        query_features: torch.Tensor,      # (batch, seq, hidden) - current stock's temporal features
        query_stock_ids: torch.Tensor,     # (batch,) - current stock IDs
        batch_stock_ids: torch.Tensor = None,  # (batch, num_stocks_in_batch) - other stocks in batch (optional, for efficiency)
        batch_stock_features: torch.Tensor = None,  # (batch, num_stocks_in_batch, hidden) - their features (optional)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attend to related stocks using learned embeddings.

        Two modes:
        1. Full attention: Attend over all stock embeddings (expensive but complete)
        2. Batch attention: Attend only to other stocks in the batch (efficient for training)

        Returns:
            output: (batch, seq, hidden) - enriched features
            attention_weights: (batch, num_heads, top_k) - which stocks attended to
        """
        batch_size, seq_len, _ = query_features.shape

        # Get query stock embeddings
        query_stock_embed = self.stock_embeddings(query_stock_ids)  # (batch, hidden)

        # Pool temporal features for relationship computation
        query_pooled = query_features.mean(dim=1)  # (batch, hidden)

        # Combine temporal + identity for query
        query_combined = query_pooled + query_stock_embed

        if batch_stock_features is not None and batch_stock_ids is not None:
            # Efficient batch mode: only attend to stocks in current batch
            num_batch_stocks = batch_stock_features.shape[1]

            # Get embeddings for batch stocks
            batch_stock_embeds = self.stock_embeddings(batch_stock_ids.view(-1))
            batch_stock_embeds = batch_stock_embeds.view(batch_size, num_batch_stocks, -1)

            # Combine with their temporal features
            key_values = batch_stock_features + batch_stock_embeds  # (batch, num_batch_stocks, hidden)

        else:
            # Full mode: attend to all stock embeddings
            all_stock_embeds = self.stock_embeddings.get_all_embeddings()  # (num_stocks, hidden)
            key_values = all_stock_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        num_kv = key_values.shape[1]

        # Project queries, keys, values
        # Query from each timestep
        q = self.q_proj(query_features)  # (batch, seq, hidden)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Keys and values from stock embeddings
        k = self.k_proj(key_values)  # (batch, num_kv, hidden)
        v = self.v_proj(key_values)

        k = k.view(batch_size, num_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # q: (batch, heads, seq, head_dim)
        # k: (batch, heads, num_kv, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Relationship gating: modulate attention per head based on query-key relationship
        # This learns different relationship types (sector, competitor, etc.)
        relationship_input = torch.cat([
            query_combined.unsqueeze(1).expand(-1, num_kv, -1),
            key_values
        ], dim=-1)  # (batch, num_kv, hidden*2)

        rel_gates = self.relationship_gate(relationship_input)  # (batch, num_kv, num_heads)
        rel_gates = rel_gates.transpose(1, 2).unsqueeze(2)  # (batch, num_heads, 1, num_kv)

        # Apply relationship gating to scores
        scores = scores * rel_gates

        # Optional: top-k attention for efficiency (only attend to most relevant stocks)
        if self.top_k < num_kv:
            # Get top-k scores for each query position
            top_scores, top_indices = scores.topk(self.top_k, dim=-1)

            # Create sparse attention
            attn_weights = F.softmax(top_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Gather values for top-k
            # Expand indices for gathering
            top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
            v_expanded = v.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
            top_v = torch.gather(v_expanded, 3, top_indices_expanded)

            # Weighted sum
            context = torch.einsum('bhsk,bhskd->bhsd', attn_weights, top_v)

            # Store which stocks were attended to (for interpretability)
            attended_stocks = top_indices[:, :, -1, :]  # Last timestep attention
        else:
            # Full attention
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context = torch.matmul(attn_weights, v)
            attended_stocks = None

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(context)

        # Residual + LayerNorm
        output = self.layer_norm(query_features + self.dropout(output))

        return output, attn_weights if attended_stocks is None else (attn_weights, attended_stocks)


class CrossStockAttention(nn.Module):
    """
    LEGACY: Explicit cross-stock attention (requires pre-computed related stocks)

    Kept for backward compatibility. Use LearnedCrossStockAttention for new training.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        max_related: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_related = max_related

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Learnable relationship type embeddings
        # Types: same_sector, competitor, supplier, customer, correlated, anchor
        self.relationship_embed = nn.Embedding(6, self.head_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        query_stock: torch.Tensor,           # (batch, seq, hidden)
        related_stocks: torch.Tensor,        # (batch, num_related, seq, hidden)
        relationship_types: torch.Tensor,    # (batch, num_related) - relationship type indices
        correlation_weights: torch.Tensor,   # (batch, num_related) - correlation strengths
        mask: torch.Tensor = None            # (batch, num_related) - valid related stocks
    ) -> torch.Tensor:

        batch_size, seq_len, _ = query_stock.shape
        num_related = related_stocks.shape[1]

        # Project query from target stock
        q = self.q_proj(query_stock)  # (batch, seq, hidden)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Project keys and values from related stocks
        # Aggregate related stock sequences first (use last hidden state or mean)
        related_repr = related_stocks.mean(dim=2)  # (batch, num_related, hidden)

        k = self.k_proj(related_repr)  # (batch, num_related, hidden)
        v = self.v_proj(related_repr)

        k = k.view(batch_size, num_related, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_related, self.num_heads, self.head_dim).transpose(1, 2)

        # Add relationship type information to keys
        rel_embed = self.relationship_embed(relationship_types)  # (batch, num_related, head_dim)
        rel_embed = rel_embed.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        k = k + rel_embed

        # Attention scores
        # q: (batch, heads, seq, head_dim)
        # k: (batch, heads, num_related, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply correlation weights as attention bias
        corr_bias = correlation_weights.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, num_related)
        scores = scores + corr_bias

        # Mask invalid related stocks
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, num_related)
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # attn_weights: (batch, heads, seq, num_related)
        # v: (batch, heads, num_related, head_dim)
        context = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(context)

        # Residual connection
        output = self.layer_norm(query_stock + self.dropout(output))

        return output


class SjoniAttention(nn.Module):
    """Multi-head self-attention with RoPE"""

    def __init__(self, config: SjoniConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.rotary = RotaryPositionalEmbedding(self.head_dim, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary(x, seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.out_proj(context)


class SjoniBlock(nn.Module):
    """Transformer block with GRN-style FFN and gradient checkpointing support"""

    def __init__(self, config: SjoniConfig):
        super().__init__()

        self.attention = SjoniAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def _forward_attention(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.attention(self.attention_norm(x), attention_mask)

    def _forward_ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ffn_norm(x))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Pre-norm architecture with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            attn_out = checkpoint(self._forward_attention, x, attention_mask, use_reentrant=False)
            x = x + attn_out
            ffn_out = checkpoint(self._forward_ffn, x, use_reentrant=False)
            x = x + ffn_out
        else:
            x = x + self._forward_attention(x, attention_mask)
            x = x + self._forward_ffn(x)
        return x


class SentimentEncoder(nn.Module):
    """
    Encodes daily sentiment scores from multiple sources
    Input: sentiment scores from Ollama LLM analysis

    SCALED UP for 500M model - deeper encoder with more capacity
    """

    def __init__(self, config: SjoniConfig):
        super().__init__()

        self.config = config

        # Sentiment sources: news, social, analyst, insider_sentiment, etc.
        self.source_embed = nn.Embedding(8, 64)  # 8 possible sources (was 32)

        # Deeper encoder for 500M model
        # Input: sentiment features + source embedding = num_sentiment_features + 64
        # After mean pooling, we get (batch, seq, num_sentiment_features + 64) = 8 + 64 = 72
        # But the concatenation expands sentiment to 64 dims, so input is 64 + 64 = 128
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),  # 64 (expanded sentiment) + 64 (source embed)
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.hidden_size // 4)
        )

        # Multi-scale temporal aggregation for sentiment history
        self.temporal_conv_short = nn.Conv1d(
            config.hidden_size // 4,
            config.hidden_size // 8,
            kernel_size=3,
            padding=1
        )
        self.temporal_conv_medium = nn.Conv1d(
            config.hidden_size // 4,
            config.hidden_size // 8,
            kernel_size=7,
            padding=3
        )

        # Combine multi-scale features
        self.temporal_fusion = nn.Linear(config.hidden_size // 4, config.hidden_size // 4)

    def forward(
        self,
        sentiment_scores: torch.Tensor,  # (batch, seq, num_sources)
        source_ids: torch.Tensor = None  # (num_sources,)
    ) -> torch.Tensor:
        batch_size, seq_len, num_sources = sentiment_scores.shape

        if source_ids is None:
            source_ids = torch.arange(num_sources, device=sentiment_scores.device)

        # Add source embeddings
        source_emb = self.source_embed(source_ids)  # (num_sources, 64)
        source_emb = source_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)

        # Expand sentiment for concatenation
        sentiment_expanded = sentiment_scores.unsqueeze(-1)  # (batch, seq, num_sources, 1)

        # Concat and encode per source, then aggregate
        combined = torch.cat([
            sentiment_expanded.expand(-1, -1, -1, 64),
            source_emb
        ], dim=-1)  # (batch, seq, num_sources, 65)

        # Mean pool across sources for now (could use attention later)
        combined = combined.mean(dim=2)  # (batch, seq, 65)

        # Encode
        encoded = self.encoder(combined)  # (batch, seq, hidden//4)

        # Multi-scale temporal smoothing
        encoded_t = encoded.transpose(1, 2)  # (batch, hidden//4, seq)
        short_scale = self.temporal_conv_short(encoded_t)   # (batch, hidden//8, seq)
        medium_scale = self.temporal_conv_medium(encoded_t)  # (batch, hidden//8, seq)

        # Concatenate multi-scale features
        multi_scale = torch.cat([short_scale, medium_scale], dim=1)  # (batch, hidden//4, seq)
        multi_scale = multi_scale.transpose(1, 2)  # (batch, seq, hidden//4)

        # Fuse multi-scale temporal features
        encoded = self.temporal_fusion(multi_scale)

        return encoded


class InstitutionalFlowEncoder(nn.Module):
    """
    Approximates institutional trading activity from observable signals:
    - Unusual volume patterns
    - Dark pool volume estimates
    - Options flow (put/call ratios, unusual activity)
    - Block trades

    SCALED UP for 500M model - deeper encoder with pattern recognition
    """

    def __init__(self, config: SjoniConfig):
        super().__init__()

        # Deeper encoder for 500M model
        self.encoder = nn.Sequential(
            nn.Linear(config.num_institutional_features, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.hidden_size // 4)
        )

        # Enhanced anomaly detector with more capacity
        self.anomaly_detector = nn.Sequential(
            nn.Linear(config.num_institutional_features, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 6)  # 6 anomaly types: accumulation, distribution, squeeze, unusual_options, block_trade, dark_pool
        )

        # Pattern memory for detecting recurring institutional behaviors
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size // 4,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        self.pattern_norm = nn.LayerNorm(config.hidden_size // 4)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        anomaly_scores = torch.sigmoid(self.anomaly_detector(features))

        # Self-attention to find institutional patterns across time
        if encoded.dim() == 3:  # Has sequence dimension
            attn_out, _ = self.pattern_attention(encoded, encoded, encoded)
            encoded = self.pattern_norm(encoded + attn_out)

        return encoded, anomaly_scores


class WeakHandsDetector(nn.Module):
    """
    Detects when retail is selling (weak hands) while institutions accumulate
    Key insight: Negative sentiment + retail selling + institutional buying = BUY signal

    SCALED UP for 500M model - deeper analysis with confidence scoring
    """

    def __init__(self, hidden_size: int):
        super().__init__()

        # Deeper detector for 500M model
        self.detector = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)  # Weak hands score: -1 (strong hands) to +1 (weak hands selling)
        )

        # Confidence estimator - how confident are we in the weak hands signal?
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        price_features: torch.Tensor,
        sentiment_features: torch.Tensor,
        institutional_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([
            price_features,
            sentiment_features,
            institutional_features
        ], dim=-1)

        weak_hands_score = torch.tanh(self.detector(combined))
        confidence = self.confidence(combined)

        return weak_hands_score, confidence


class MarketRegimeHead(nn.Module):
    """
    Classifies current market regime:
    - Bubble forming
    - Crash/Correction
    - Recovery
    - Normal trading
    - IPO spike phase
    - IPO decline phase

    SCALED UP for 500M model - deeper classifier with regime transition prediction
    """

    def __init__(self, config: SjoniConfig):
        super().__init__()

        # Deeper classifier for 500M model
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, config.num_regime_classes)
        )

        # Regime transition predictor - what regime is likely next?
        self.transition_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, config.num_regime_classes * config.num_regime_classes)
        )
        self.num_regimes = config.num_regime_classes

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use mean pooled representation
        pooled = hidden_states.mean(dim=1)
        regime_logits = self.classifier(pooled)

        # Transition matrix (from_regime -> to_regime probabilities)
        transition_flat = self.transition_predictor(pooled)
        transition_matrix = transition_flat.view(-1, self.num_regimes, self.num_regimes)
        transition_probs = F.softmax(transition_matrix, dim=-1)

        return regime_logits, transition_probs


class DelistingRiskHead(nn.Module):
    """
    Predicts probability of delisting based on patterns from historical delistings

    SCALED UP for 500M model - multi-horizon risk prediction
    """

    def __init__(self, config: SjoniConfig):
        super().__init__()

        # Deeper risk predictor for 500M model
        self.risk_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 4)  # 30-day, 90-day, 180-day, 1-year delisting risk
        )

        # Risk factor attribution - why is delisting likely?
        self.risk_factors = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 6)  # Financial distress, compliance issues, acquisition, going private, fraud, other
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = hidden_states.mean(dim=1)
        multi_horizon_risk = torch.sigmoid(self.risk_predictor(pooled))
        risk_factor_weights = F.softmax(self.risk_factors(pooled), dim=-1)
        return multi_horizon_risk, risk_factor_weights


class PricePredictionHead(nn.Module):
    """
    Multi-horizon price prediction with confidence intervals

    SCALED UP for 500M model - deeper prediction with multiple quantiles
    """

    def __init__(self, config: SjoniConfig):
        super().__init__()

        self.horizon = config.prediction_horizon

        # Deeper point predictor for 500M model
        self.point_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, config.prediction_horizon)  # Next N days returns
        )

        # Enhanced uncertainty estimation with attention to sequence
        self.uncertainty_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, config.prediction_horizon)
        )

        # Extended quantile predictions (5th, 10th, 25th, 50th, 75th, 90th, 95th)
        self.quantile_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, config.prediction_horizon * 7)  # 7 quantiles
        )

        # Direction confidence (probability of positive return)
        self.direction_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, config.prediction_horizon),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Use last hidden state for point prediction
        last_hidden = hidden_states[:, -1, :]

        point_pred = self.point_predictor(last_hidden)

        # Use attention for uncertainty (look at entire sequence)
        attn_out, _ = self.uncertainty_attention(hidden_states, hidden_states, hidden_states)
        uncertainty_context = attn_out[:, -1, :]  # Take last position after attention
        uncertainty = F.softplus(self.uncertainty_predictor(uncertainty_context))

        # Quantiles and direction
        quantiles = self.quantile_predictor(last_hidden).view(-1, self.horizon, 7)
        direction_prob = self.direction_predictor(last_hidden)

        return {
            'point': point_pred,
            'uncertainty': uncertainty,
            'quantiles': quantiles,  # (batch, horizon, 7)
            'direction_prob': direction_prob  # Probability of positive return
        }


class Sjoni(nn.Module):
    """
    Sjoni - 500M Parameter Market Intelligence Model

    Combines:
    - Technical price analysis (25 years of data)
    - News sentiment (via local Ollama LLM)
    - Cross-stock relationships (200 related stocks)
    - Institutional flow detection
    - Weak hands detection
    - Market regime classification
    - Delisting risk assessment

    Memory optimizations:
    - Gradient checkpointing enabled
    - Mixed precision training supported
    """

    def __init__(self, config: SjoniConfig = None):
        super().__init__()

        self.config = config or SjoniConfig()

        # === Input Encoders (SCALED UP for 500M) ===

        # Price and technical features - deeper encoder
        self.price_encoder = nn.Sequential(
            nn.Linear(self.config.num_price_features, 512),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, self.config.hidden_size // 2)
        )

        # Sentiment encoder (Ollama LLM outputs)
        self.sentiment_encoder = SentimentEncoder(self.config)

        # Institutional flow encoder
        self.institutional_encoder = InstitutionalFlowEncoder(self.config)

        # Market context encoder (VIX, sector indices, etc.) - deeper for 500M
        self.market_encoder = nn.Sequential(
            nn.Linear(self.config.num_market_features, 256),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self.config.hidden_size // 4)
        )

        # === Embeddings ===

        # Sector and industry embeddings
        self.sector_embed = nn.Embedding(self.config.num_sectors, self.config.sector_embed_dim)
        self.industry_embed = nn.Embedding(self.config.num_industries, self.config.industry_embed_dim)

        # Stock age embedding (for IPO pattern detection)
        self.age_embed = nn.Embedding(100, 32)  # 0-99 months since listing

        # === Feature Fusion ===

        # Variable selection network
        input_sizes = {
            'price': self.config.hidden_size // 2,
            'sentiment': self.config.hidden_size // 4,
            'institutional': self.config.hidden_size // 4,
            'market': self.config.hidden_size // 4,
            'sector': self.config.sector_embed_dim,
            'industry': self.config.industry_embed_dim
        }
        self.variable_selection = VariableSelectionNetwork(
            input_sizes,
            self.config.hidden_size,
            self.config.dropout
        )

        # === Cross-Stock Attention ===
        if self.config.use_learned_relationships:
            # NEW: Learned cross-stock attention (model discovers relationships)
            self.cross_stock_attention = LearnedCrossStockAttention(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.cross_stock_heads,
                num_stocks=self.config.num_stocks,
                stock_embed_dim=self.config.stock_embed_dim,
                top_k_attend=self.config.max_related_stocks,
                dropout=self.config.dropout
            )
        else:
            # LEGACY: Explicit cross-stock attention (requires pre-computed relationships)
            self.cross_stock_attention = CrossStockAttention(
                self.config.hidden_size,
                self.config.cross_stock_heads,
                self.config.max_related_stocks,
                self.config.dropout
            )

        # === Main Transformer ===
        self.transformer_blocks = nn.ModuleList([
            SjoniBlock(self.config)
            for _ in range(self.config.num_layers)
        ])

        # === Output Heads ===

        # Price prediction
        self.price_head = PricePredictionHead(self.config)

        # Market regime classification
        self.regime_head = MarketRegimeHead(self.config)

        # Delisting risk
        self.delisting_head = DelistingRiskHead(self.config)

        # Weak hands detector
        self.weak_hands_detector = WeakHandsDetector(
            self.config.hidden_size // 2 +
            self.config.hidden_size // 4 +
            self.config.hidden_size // 4
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(self.config.hidden_size)

        # Initialize weights
        self.apply(self._init_weights)

        # Print parameter count
        self._print_params()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Sjoni Parameters: {total:,} total, {trainable:,} trainable")

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self._gradient_checkpointing = True

        # Enable for transformer blocks
        for block in self.transformer_blocks:
            block.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self._gradient_checkpointing = False

        for block in self.transformer_blocks:
            block.gradient_checkpointing = False

    def forward(
        self,
        price_features: torch.Tensor,           # (batch, seq, num_price_features)
        sentiment_features: torch.Tensor,        # (batch, seq, num_sentiment_features)
        institutional_features: torch.Tensor,    # (batch, seq, num_institutional_features)
        market_features: torch.Tensor,           # (batch, seq, num_market_features)
        sector_ids: torch.Tensor,                # (batch,) sector index
        industry_ids: torch.Tensor,              # (batch,) industry index
        stock_age_months: torch.Tensor,          # (batch,) months since listing
        stock_ids: torch.Tensor = None,          # (batch,) stock index for learned relationships
        batch_stock_ids: torch.Tensor = None,    # (batch, num_stocks_in_batch) other stocks in batch
        batch_stock_features: torch.Tensor = None,  # (batch, num_stocks_in_batch, hidden) their features
        # Legacy explicit relationships (only used if use_learned_relationships=False)
        related_stock_features: torch.Tensor = None,    # (batch, num_related, seq, hidden)
        relationship_types: torch.Tensor = None,        # (batch, num_related)
        correlation_weights: torch.Tensor = None,       # (batch, num_related)
        related_mask: torch.Tensor = None,              # (batch, num_related)
        attention_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:

        batch_size, seq_len, _ = price_features.shape

        # === Encode Inputs ===

        price_encoded = self.price_encoder(price_features)  # (batch, seq, hidden/2)
        sentiment_encoded = self.sentiment_encoder(sentiment_features)  # (batch, seq, hidden/4)
        institutional_encoded, anomaly_scores = self.institutional_encoder(institutional_features)
        market_encoded = self.market_encoder(market_features)  # (batch, seq, hidden/4)

        # Get embeddings and expand to sequence length
        sector_emb = self.sector_embed(sector_ids).unsqueeze(1).expand(-1, seq_len, -1)
        industry_emb = self.industry_embed(industry_ids).unsqueeze(1).expand(-1, seq_len, -1)
        age_emb = self.age_embed(stock_age_months.clamp(0, 99)).unsqueeze(1).expand(-1, seq_len, -1)

        # === Variable Selection ===

        inputs = {
            'price': price_encoded,
            'sentiment': sentiment_encoded,
            'institutional': institutional_encoded,
            'market': market_encoded,
            'sector': sector_emb,
            'industry': industry_emb
        }

        fused, selection_weights = self.variable_selection(inputs)

        # === Cross-Stock Attention ===
        cross_stock_attn_weights = None

        if self.config.use_learned_relationships:
            # NEW: Learned cross-stock attention
            if stock_ids is not None:
                fused, cross_stock_attn_weights = self.cross_stock_attention(
                    query_features=fused,
                    query_stock_ids=stock_ids,
                    batch_stock_ids=batch_stock_ids,
                    batch_stock_features=batch_stock_features
                )
        else:
            # LEGACY: Explicit cross-stock attention
            if related_stock_features is not None:
                fused = self.cross_stock_attention(
                    fused,
                    related_stock_features,
                    relationship_types,
                    correlation_weights,
                    related_mask
                )

        # === Main Transformer ===

        hidden = fused
        for block in self.transformer_blocks:
            hidden = block(hidden, attention_mask)

        hidden = self.final_norm(hidden)

        # === Output Heads (SCALED UP for 500M) ===

        # Price predictions (now includes direction probability)
        price_output = self.price_head(hidden)

        # Market regime (now includes transition probabilities)
        regime_logits, regime_transitions = self.regime_head(hidden)

        # Delisting risk (now multi-horizon with risk factors)
        delisting_risk, delisting_factors = self.delisting_head(hidden)

        # Weak hands detection (now includes confidence)
        weak_hands_score, weak_hands_confidence = self.weak_hands_detector(
            price_encoded[:, -1, :],
            sentiment_encoded[:, -1, :],
            institutional_encoded[:, -1, :]
        )

        return {
            # Price predictions
            'price_prediction': price_output['point'],
            'price_uncertainty': price_output['uncertainty'],
            'price_quantiles': price_output['quantiles'],
            'price_direction_prob': price_output['direction_prob'],

            # Market regime
            'regime_logits': regime_logits,
            'regime_transitions': regime_transitions,

            # Delisting risk
            'delisting_risk': delisting_risk,  # (batch, 4) - 30/90/180/365 day risk
            'delisting_factors': delisting_factors,  # (batch, 6) - factor attribution

            # Weak hands
            'weak_hands_score': weak_hands_score,
            'weak_hands_confidence': weak_hands_confidence,

            # Institutional anomalies
            'anomaly_scores': anomaly_scores,  # (batch, seq, 6) - 6 anomaly types

            # Interpretability
            'selection_weights': selection_weights,
            'cross_stock_attention': cross_stock_attn_weights,
            'hidden_states': hidden
        }

    @torch.no_grad()
    def predict(
        self,
        price_features: torch.Tensor,
        sentiment_features: torch.Tensor,
        institutional_features: torch.Tensor,
        market_features: torch.Tensor,
        sector_ids: torch.Tensor,
        industry_ids: torch.Tensor,
        stock_age_months: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Inference mode prediction"""
        self.eval()
        return self.forward(
            price_features,
            sentiment_features,
            institutional_features,
            market_features,
            sector_ids,
            industry_ids,
            stock_age_months,
            **kwargs
        )


def create_sjoni_500m() -> Sjoni:
    """
    Create the 500M parameter MarketBrain model (default)

    Memory requirements:
    - ~2GB for model parameters (FP16)
    - ~4GB for gradients (FP16)
    - ~6-8GB for activations with gradient checkpointing
    - Total: ~12-14GB VRAM for training

    Fits comfortably on RTX 4090 (24GB) with mixed precision.
    """

    config = SjoniConfig(
        # Core dimensions - SCALED for 500M
        hidden_size=1280,
        num_layers=24,
        num_heads=16,
        intermediate_size=5120,
        dropout=0.1,

        # Sequence settings
        max_seq_length=252,
        prediction_horizon=5,

        # Input features
        num_price_features=32,
        num_sentiment_features=20,  # 17 from PriceDerivedSentiment + 3 padding
        num_institutional_features=12,
        num_market_features=16,

        # Embeddings - SCALED
        num_sectors=11,
        num_industries=69,
        sector_embed_dim=128,
        industry_embed_dim=256,

        # Cross-stock attention - SCALED
        max_related_stocks=200,
        cross_stock_heads=32,

        # Output
        num_regime_classes=6,

        # Memory optimization
        use_gradient_checkpointing=True,
        use_flash_attention=True
    )

    model = Sjoni(config)

    # Enable gradient checkpointing if configured
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def create_sjoni_100m() -> Sjoni:
    """Create the smaller 100M parameter MarketBrain model (for testing/comparison)"""

    config = SjoniConfig(
        hidden_size=768,
        num_layers=16,
        num_heads=12,
        intermediate_size=3072,
        dropout=0.1,
        max_seq_length=252,
        prediction_horizon=5,
        num_price_features=32,
        num_sentiment_features=20,  # 17 from PriceDerivedSentiment + 3 padding
        num_institutional_features=12,
        num_market_features=16,
        num_sectors=11,
        num_industries=69,
        sector_embed_dim=64,
        industry_embed_dim=128,
        max_related_stocks=50,
        cross_stock_heads=8,
        num_regime_classes=6,
        use_gradient_checkpointing=False,
        use_flash_attention=False
    )

    return Sjoni(config)


# Alias for default model
create_sjoni = create_sjoni_500m


if __name__ == "__main__":
    print("=" * 60)
    print("Sjoni - 500M Parameter Model Test")
    print("=" * 60)

    # Test 500M model creation
    print("\nCreating 500M parameter model...")
    model = create_sjoni_500m()

    # Test forward pass
    batch_size = 2
    seq_len = 120

    dummy_input = {
        'price_features': torch.randn(batch_size, seq_len, 32),
        'sentiment_features': torch.randn(batch_size, seq_len, 8),
        'institutional_features': torch.randn(batch_size, seq_len, 12),
        'market_features': torch.randn(batch_size, seq_len, 16),
        'sector_ids': torch.randint(0, 11, (batch_size,)),
        'industry_ids': torch.randint(0, 69, (batch_size,)),
        'stock_age_months': torch.randint(0, 100, (batch_size,)),
        'stock_ids': torch.randint(0, 6000, (batch_size,)),  # For learned cross-stock attention
    }

    print("\nRunning forward pass...")
    output = model(**dummy_input)

    print(f"\nOutput shapes:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Memory estimation
    print("\n" + "=" * 60)
    print("Memory Estimation (FP16 Training):")
    print("=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    param_memory_gb = (total_params * 2) / (1024**3)  # FP16 = 2 bytes
    grad_memory_gb = param_memory_gb  # Gradients same size as params
    optimizer_memory_gb = param_memory_gb * 2  # Adam has 2 states per param

    print(f"  Parameters: {param_memory_gb:.2f} GB")
    print(f"  Gradients:  {grad_memory_gb:.2f} GB")
    print(f"  Optimizer:  {optimizer_memory_gb:.2f} GB")
    print(f"  Estimated activations: ~4-6 GB (with gradient checkpointing)")
    print(f"  Total: ~{param_memory_gb + grad_memory_gb + optimizer_memory_gb + 5:.1f} GB")
    fits = param_memory_gb + grad_memory_gb + optimizer_memory_gb + 5 < 24
    print(f"\n  RTX 4090 (24GB): {'FITS' if fits else 'TOO LARGE'}")
