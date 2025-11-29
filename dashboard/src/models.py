"""
ArgusTrader Data Models
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class Trade:
    """Real-time trade tick."""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    exchange: str = ""
    conditions: list[int] = field(default_factory=list)


@dataclass
class Quote:
    """Real-time quote (bid/ask)."""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2


@dataclass
class Bar:
    """OHLCV bar (candlestick)."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0
    trade_count: int = 0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def body(self) -> float:
        return self.close - self.open

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    unrealized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    def update_pnl(self, current_price: float) -> None:
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.avg_cost) * self.quantity


@dataclass
class Signal:
    """Trading signal from strategy."""
    symbol: str
    timestamp: datetime
    signal_type: str  # "BUY", "SELL", "HOLD"
    strength: float  # 0.0 to 1.0
    source: str  # "technical", "ml", "sentiment", "combined"
    metadata: dict = field(default_factory=dict)


class MarketDataStore:
    """In-memory store for real-time market data."""

    def __init__(self, max_bars: int = 500):
        self.max_bars = max_bars
        self._trades: dict[str, list[Trade]] = {}
        self._quotes: dict[str, Quote] = {}
        self._bars: dict[str, pd.DataFrame] = {}
        self._positions: dict[str, Position] = {}
        self._signals: list[Signal] = []

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the store."""
        if trade.symbol not in self._trades:
            self._trades[trade.symbol] = []
        self._trades[trade.symbol].append(trade)
        # Keep only recent trades
        if len(self._trades[trade.symbol]) > 1000:
            self._trades[trade.symbol] = self._trades[trade.symbol][-500:]

    def update_quote(self, quote: Quote) -> None:
        """Update the latest quote for a symbol."""
        self._quotes[quote.symbol] = quote

    def add_bar(self, bar: Bar) -> None:
        """Add an OHLCV bar to the store."""
        if bar.symbol not in self._bars:
            self._bars[bar.symbol] = pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"]
            )

        new_row = pd.DataFrame([{
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap,
        }])

        self._bars[bar.symbol] = pd.concat(
            [self._bars[bar.symbol], new_row], ignore_index=True
        )

        # Keep only recent bars
        if len(self._bars[bar.symbol]) > self.max_bars:
            self._bars[bar.symbol] = self._bars[bar.symbol].tail(self.max_bars)

    def get_bars(self, symbol: str) -> pd.DataFrame:
        """Get bars DataFrame for a symbol."""
        if symbol not in self._bars:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"]
            )
        return self._bars[symbol].copy()

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote for a symbol."""
        return self._quotes.get(symbol)

    def get_recent_trades(self, symbol: str, n: int = 100) -> list[Trade]:
        """Get recent trades for a symbol."""
        if symbol not in self._trades:
            return []
        return self._trades[symbol][-n:]

    def update_position(self, position: Position) -> None:
        """Update position for a symbol."""
        self._positions[position.symbol] = position

    def get_positions(self) -> dict[str, Position]:
        """Get all positions."""
        return self._positions.copy()

    def add_signal(self, signal: Signal) -> None:
        """Add a trading signal."""
        self._signals.append(signal)
        # Keep only recent signals
        if len(self._signals) > 100:
            self._signals = self._signals[-50:]

    def get_signals(self, n: int = 10) -> list[Signal]:
        """Get recent signals."""
        return self._signals[-n:]

    @property
    def symbols(self) -> list[str]:
        """Get all tracked symbols."""
        return list(set(self._bars.keys()) | set(self._quotes.keys()))


# Global data store instance
data_store = MarketDataStore()
