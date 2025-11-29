# Sjoni - The Stock Predictor

*Old Norse: Sjóni (ᛋᛃᚬᚾᛁ) = "The Seer"*

A **547M parameter** Temporal Fusion Transformer for market intelligence. Sjoni sees patterns that others miss by combining 26 years of market data with learned cross-stock relationships.

---

## Features

### Model Architecture
- **547M parameters** - Scaled Temporal Fusion Transformer
- **Learned Cross-Stock Attention** - Model discovers stock relationships during training (no hardcoded correlations)
- **200 related stocks** with 32 attention heads for cross-correlation
- **6,000 stock embeddings** - Each stock has a learned identity vector
- Gradient checkpointing for RTX 4090 (24GB VRAM)

### Data Pipeline
- **26 years of history** (1999-present) for 5,000+ NASDAQ stocks
- **Price-derived sentiment** - Extracts "fossilized" sentiment from gaps, volume, reversals
- **Institutional flow detection** - Smart money vs weak hands analysis
- **GICS sector/industry classification** - 11 sectors, 69 industries
- **Delisted stock data** - Eliminates survivorship bias

### Prediction Outputs
| Output | Description |
|--------|-------------|
| `price_prediction` | 5-day forward returns |
| `price_uncertainty` | Confidence intervals per prediction |
| `price_direction_prob` | Probability of positive return |
| `regime_logits` | Market regime (Bubble/Crash/Recovery/Normal/IPO) |
| `delisting_risk` | 30/90/180/365-day delisting probability |
| `weak_hands_score` | Retail panic indicator (-1 to +1) |

---

## Quick Start

### Requirements
- Python 3.10+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU with 12GB+ VRAM (24GB recommended)

### Installation

```bash
git clone https://github.com/Deverydoo/SjoniTheStockPredicter.git
cd SjoniTheStockPredicter

# Create environment
conda create -n sjoni python=3.10
conda activate sjoni

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy yfinance scikit-learn tqdm
```

### Fetch Historical Data

```bash
cd training

# Fetch all NASDAQ symbols with max history (1999-present)
python fetch_historical.py --symbols all --start 1999-01-01 --workers 10

# Fetch market context (VIX, indices, etc.)
python fetch_historical.py --symbols bluechip --market-context --workers 10
```

### Train the Model

```bash
# Full training
python train_sjoni.py

# Quick test (2 epochs, small batch)
python train_sjoni.py --test

# Resume from checkpoint
python train_sjoni.py --resume models/sjoni_latest.pt
```

### Inference

```python
from training.src.sjoni import create_sjoni_500m
import torch

# Load model
model = create_sjoni_500m()
model.load_state_dict(torch.load('models/sjoni_best.pt')['model_state_dict'])
model.eval()

# Prepare your features (see training pipeline for format)
outputs = model.predict(
    price_features=price_tensor,
    sentiment_features=sentiment_tensor,
    institutional_features=inst_tensor,
    market_features=market_tensor,
    sector_ids=sector_ids,
    industry_ids=industry_ids,
    stock_age_months=age_tensor,
    stock_ids=stock_ids
)

# Get predictions
predicted_returns = outputs['price_prediction']  # (batch, 5) - next 5 days
direction_prob = outputs['price_direction_prob']  # (batch, 5) - P(return > 0)
uncertainty = outputs['price_uncertainty']  # (batch, 5) - confidence
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SJONI 547M                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │    Price     │  │  Sentiment   │  │ Institutional│  │   Market     │ │
│  │   Encoder    │  │   Encoder    │  │    Flow      │  │   Context    │ │
│  │   (32→640)   │  │   (8→320)    │  │   (12→320)   │  │   (16→320)   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                 │                 │          │
│         └─────────────────┼─────────────────┼─────────────────┘          │
│                           │                 │                            │
│                    ┌──────▼─────────────────▼──────┐                     │
│                    │   Variable Selection Network   │                     │
│                    │        (learns feature         │                     │
│                    │         importance)            │                     │
│                    └──────────────┬─────────────────┘                     │
│                                   │                                       │
│                    ┌──────────────▼─────────────────┐                     │
│                    │   Learned Cross-Stock Attention │                    │
│                    │   (200 stocks, 32 heads)        │                    │
│                    │   Discovers relationships       │                    │
│                    └──────────────┬─────────────────┘                     │
│                                   │                                       │
│                    ┌──────────────▼─────────────────┐                     │
│                    │   24-Layer Transformer          │                    │
│                    │   (1280 hidden, 16 heads)       │                    │
│                    │   RoPE positional encoding      │                    │
│                    └──────────────┬─────────────────┘                     │
│                                   │                                       │
│         ┌─────────────────────────┼─────────────────────────┐            │
│         │                         │                         │            │
│  ┌──────▼──────┐          ┌───────▼───────┐         ┌───────▼───────┐   │
│  │   Price     │          │    Regime     │         │   Delisting   │   │
│  │   Head      │          │     Head      │         │     Head      │   │
│  │  (returns,  │          │  (6 classes)  │         │ (4 horizons)  │   │
│  │ uncertainty)│          │               │         │               │   │
│  └─────────────┘          └───────────────┘         └───────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Training Data

### Data Sources (All Free)
| Source | Coverage | Purpose |
|--------|----------|---------|
| Yahoo Finance | 1999-present | OHLCV price data |
| GDELT GKG | 2015-present | News sentiment |
| Price-derived | 1999-2015 | Sentiment proxy from price action |
| Synthetic | N/A | Delisting patterns for survivorship bias |

### Feature Categories

**Price Features (32)**
- Returns, log returns, momentum (1/5/10/20 day)
- Moving averages (SMA/EMA 5/10/20/50/200)
- Bollinger Bands, RSI, MACD
- ATR, volatility, volume ratios

**Sentiment Features (8)**
- Overnight gap, close position, volume sentiment
- Fear signal, bullish/bearish volume
- Composite sentiment proxy

**Institutional Features (12)**
- Unusual volume, block trade probability
- Accumulation/distribution, money flow index
- Smart money flow, weak hands score
- Short squeeze potential

---

## Hardware Requirements

| Configuration | VRAM | Batch Size | Training Speed |
|--------------|------|------------|----------------|
| RTX 4090 | 24GB | 32 | ~2 hours/epoch |
| RTX 3090 | 24GB | 32 | ~3 hours/epoch |
| RTX 3080 | 10GB | 8 | ~6 hours/epoch |

Mixed precision (FP16) and gradient checkpointing are enabled by default.

---

## Project Structure

```
SjoniTheStockPredicter/
├── training/
│   ├── src/
│   │   ├── sjoni.py              # 547M model architecture
│   │   ├── features.py           # Technical indicators
│   │   ├── gdelt_historical.py   # Sentiment pipeline
│   │   ├── institutional_flow.py # Smart money detection
│   │   ├── sector_classifier.py  # GICS classification
│   │   └── delisted_fetcher.py   # Survivorship bias data
│   ├── train_sjoni.py            # Training pipeline
│   ├── fetch_historical.py       # Data fetcher
│   └── data/                     # (gitignored) Training data
├── engine/                       # C++ inference engine (future)
├── LICENSE                       # BSL 1.1
└── README.md
```

---

## License

**Business Source License 1.1** (BSL)

### Permitted Uses
- Personal trading and research
- Academic research and education
- Internal organizational use
- Derivative works (under same license)
- Consulting (client executes trades independently)

### Restricted Uses
- Commercial automated trading services (executing trades for third parties for compensation)

The license converts to **GPL v3** four years after each release.

See [LICENSE](LICENSE) for full terms.

---

## Disclaimer

This software is for educational and research purposes. It is **not financial advice**.

- Past performance does not guarantee future results
- The model may produce inaccurate predictions
- Always consult qualified financial professionals
- Trade at your own risk

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## Acknowledgments

- Temporal Fusion Transformer paper by Google Research
- GDELT Project for free news data
- Yahoo Finance for historical price data

---

*Built with PyTorch and Viking wisdom*
