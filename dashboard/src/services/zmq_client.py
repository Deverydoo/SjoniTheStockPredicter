"""
ZeroMQ Client for receiving market data from C++ engine.

The C++ engine publishes market data on different topics:
- "TRADE" - Real-time trade ticks
- "QUOTE" - Real-time quotes (bid/ask)
- "BAR"   - OHLCV bars (1-minute aggregates)
- "SIGNAL" - Trading signals from strategy engine
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty
from typing import Callable, Optional

import zmq

logger = logging.getLogger(__name__)


@dataclass
class TradeMessage:
    """Trade message from C++ engine."""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    exchange: str = ""


@dataclass
class QuoteMessage:
    """Quote message from C++ engine."""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int


@dataclass
class BarMessage:
    """Bar (OHLCV) message from C++ engine."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0


@dataclass
class SignalMessage:
    """Trading signal from strategy engine."""
    symbol: str
    timestamp: datetime
    signal_type: str  # BUY, SELL, HOLD
    strength: float
    source: str


class ZMQClient:
    """
    ZeroMQ subscriber client for receiving market data.

    Uses PUB/SUB pattern:
    - C++ engine publishes on tcp://localhost:5555
    - Dashboard subscribes to specific topics
    """

    def __init__(
        self,
        address: str = "tcp://127.0.0.1:5555",
        topics: Optional[list[str]] = None,
    ):
        self.address = address
        self.topics = topics or ["TRADE", "QUOTE", "BAR", "SIGNAL"]

        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Message queues for each type
        self.trade_queue: Queue[TradeMessage] = Queue(maxsize=10000)
        self.quote_queue: Queue[QuoteMessage] = Queue(maxsize=10000)
        self.bar_queue: Queue[BarMessage] = Queue(maxsize=1000)
        self.signal_queue: Queue[SignalMessage] = Queue(maxsize=100)

        # Callbacks
        self._on_trade: Optional[Callable[[TradeMessage], None]] = None
        self._on_quote: Optional[Callable[[QuoteMessage], None]] = None
        self._on_bar: Optional[Callable[[BarMessage], None]] = None
        self._on_signal: Optional[Callable[[SignalMessage], None]] = None

        # Stats
        self.messages_received = 0
        self.last_message_time: Optional[datetime] = None

    def on_trade(self, callback: Callable[[TradeMessage], None]) -> None:
        """Set callback for trade messages."""
        self._on_trade = callback

    def on_quote(self, callback: Callable[[QuoteMessage], None]) -> None:
        """Set callback for quote messages."""
        self._on_quote = callback

    def on_bar(self, callback: Callable[[BarMessage], None]) -> None:
        """Set callback for bar messages."""
        self._on_bar = callback

    def on_signal(self, callback: Callable[[SignalMessage], None]) -> None:
        """Set callback for signal messages."""
        self._on_signal = callback

    def connect(self) -> bool:
        """Connect to the ZMQ publisher."""
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.connect(self.address)

            # Subscribe to topics
            for topic in self.topics:
                self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
                logger.info(f"Subscribed to topic: {topic}")

            # Set receive timeout
            self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout

            logger.info(f"Connected to ZMQ publisher at {self.address}")
            return True

        except zmq.ZMQError as e:
            logger.error(f"Failed to connect to ZMQ: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from ZMQ publisher."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None
        logger.info("Disconnected from ZMQ publisher")

    def start(self) -> None:
        """Start the receiver thread."""
        if self._running:
            return

        if not self._socket:
            if not self.connect():
                return

        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info("ZMQ receiver started")

    def stop(self) -> None:
        """Stop the receiver thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.disconnect()
        logger.info("ZMQ receiver stopped")

    def is_connected(self) -> bool:
        """Check if connected and receiving data."""
        if not self._socket:
            return False
        # Consider connected if we received a message in the last 30 seconds
        if self.last_message_time:
            elapsed = (datetime.now() - self.last_message_time).total_seconds()
            return elapsed < 30
        return False

    def _receive_loop(self) -> None:
        """Main receive loop running in background thread."""
        while self._running:
            try:
                # Receive multipart message: [topic, json_data]
                message = self._socket.recv_multipart()

                if len(message) >= 2:
                    topic = message[0].decode("utf-8")
                    data = json.loads(message[1].decode("utf-8"))

                    self.messages_received += 1
                    self.last_message_time = datetime.now()

                    self._process_message(topic, data)

            except zmq.Again:
                # Timeout, no message received
                continue
            except zmq.ZMQError as e:
                if self._running:
                    logger.error(f"ZMQ receive error: {e}")
                break
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in message: {e}")
            except Exception as e:
                logger.exception(f"Error processing message: {e}")

    def _process_message(self, topic: str, data: dict) -> None:
        """Process a received message based on its topic."""
        try:
            if topic == "TRADE":
                msg = TradeMessage(
                    symbol=data.get("symbol", ""),
                    timestamp=datetime.fromtimestamp(data.get("timestamp", 0) / 1e9),
                    price=data.get("price", 0.0),
                    size=data.get("size", 0),
                    exchange=data.get("exchange", ""),
                )
                self.trade_queue.put_nowait(msg)
                if self._on_trade:
                    self._on_trade(msg)

            elif topic == "QUOTE":
                msg = QuoteMessage(
                    symbol=data.get("symbol", ""),
                    timestamp=datetime.fromtimestamp(data.get("timestamp", 0) / 1e9),
                    bid_price=data.get("bid_price", 0.0),
                    ask_price=data.get("ask_price", 0.0),
                    bid_size=data.get("bid_size", 0),
                    ask_size=data.get("ask_size", 0),
                )
                self.quote_queue.put_nowait(msg)
                if self._on_quote:
                    self._on_quote(msg)

            elif topic == "BAR":
                msg = BarMessage(
                    symbol=data.get("symbol", ""),
                    timestamp=datetime.fromtimestamp(data.get("timestamp", 0) / 1e9),
                    open=data.get("open", 0.0),
                    high=data.get("high", 0.0),
                    low=data.get("low", 0.0),
                    close=data.get("close", 0.0),
                    volume=data.get("volume", 0),
                    vwap=data.get("vwap", 0.0),
                )
                self.bar_queue.put_nowait(msg)
                if self._on_bar:
                    self._on_bar(msg)

            elif topic == "SIGNAL":
                msg = SignalMessage(
                    symbol=data.get("symbol", ""),
                    timestamp=datetime.fromtimestamp(data.get("timestamp", 0) / 1e9),
                    signal_type=data.get("signal_type", "HOLD"),
                    strength=data.get("strength", 0.0),
                    source=data.get("source", ""),
                )
                self.signal_queue.put_nowait(msg)
                if self._on_signal:
                    self._on_signal(msg)

        except Exception as e:
            logger.warning(f"Failed to process {topic} message: {e}")

    def get_trades(self, max_items: int = 100) -> list[TradeMessage]:
        """Get recent trades from queue."""
        trades = []
        for _ in range(max_items):
            try:
                trades.append(self.trade_queue.get_nowait())
            except Empty:
                break
        return trades

    def get_bars(self, max_items: int = 100) -> list[BarMessage]:
        """Get recent bars from queue."""
        bars = []
        for _ in range(max_items):
            try:
                bars.append(self.bar_queue.get_nowait())
            except Empty:
                break
        return bars


class MarketDataReceiver:
    """
    Higher-level interface for receiving and storing market data.
    Integrates with the dashboard's data store.
    """

    def __init__(self, zmq_address: str = "tcp://127.0.0.1:5555"):
        self.client = ZMQClient(address=zmq_address)
        self._data_lock = threading.Lock()

        # Store latest data
        self._latest_quotes: dict[str, QuoteMessage] = {}
        self._bars: dict[str, list[BarMessage]] = {}
        self._signals: list[SignalMessage] = []

        # Set up callbacks
        self.client.on_quote(self._handle_quote)
        self.client.on_bar(self._handle_bar)
        self.client.on_signal(self._handle_signal)

    def start(self) -> None:
        """Start receiving market data."""
        self.client.start()

    def stop(self) -> None:
        """Stop receiving market data."""
        self.client.stop()

    def is_connected(self) -> bool:
        """Check connection status."""
        return self.client.is_connected()

    def _handle_quote(self, quote: QuoteMessage) -> None:
        """Store latest quote."""
        with self._data_lock:
            self._latest_quotes[quote.symbol] = quote

    def _handle_bar(self, bar: BarMessage) -> None:
        """Store bar data."""
        with self._data_lock:
            if bar.symbol not in self._bars:
                self._bars[bar.symbol] = []
            self._bars[bar.symbol].append(bar)
            # Keep last 500 bars
            if len(self._bars[bar.symbol]) > 500:
                self._bars[bar.symbol] = self._bars[bar.symbol][-500:]

    def _handle_signal(self, signal: SignalMessage) -> None:
        """Store signal."""
        with self._data_lock:
            self._signals.append(signal)
            # Keep last 100 signals
            if len(self._signals) > 100:
                self._signals = self._signals[-100:]

    def get_quote(self, symbol: str) -> Optional[QuoteMessage]:
        """Get latest quote for symbol."""
        with self._data_lock:
            return self._latest_quotes.get(symbol)

    def get_bars(self, symbol: str) -> list[BarMessage]:
        """Get bars for symbol."""
        with self._data_lock:
            return list(self._bars.get(symbol, []))

    def get_signals(self, n: int = 10) -> list[SignalMessage]:
        """Get recent signals."""
        with self._data_lock:
            return list(self._signals[-n:])

    def get_stats(self) -> dict:
        """Get receiver statistics."""
        return {
            "messages_received": self.client.messages_received,
            "last_message_time": self.client.last_message_time,
            "is_connected": self.is_connected(),
            "symbols_tracked": list(self._bars.keys()),
        }
