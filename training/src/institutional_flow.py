"""
Institutional Flow Detection Module

Approximates institutional trading activity from publicly available data:
- Unusual volume analysis
- Block trade detection
- Options flow patterns (put/call ratios)
- Dark pool volume estimates
- Short interest analysis
- Insider trading signals

This module creates features for the MarketBrain model to detect
"weak hands" vs "strong hands" market dynamics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yfinance as yf
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InstitutionalSignals:
    """Container for institutional flow signals"""
    unusual_volume_score: float      # 0-1, higher = more unusual volume
    block_trade_indicator: float     # 0-1, probability of block trades
    accumulation_score: float        # -1 to 1, positive = accumulation
    distribution_score: float        # -1 to 1, positive = distribution
    smart_money_flow: float          # -1 to 1, estimated institutional direction
    retail_flow: float               # -1 to 1, estimated retail direction
    weak_hands_score: float          # -1 to 1, higher = more retail panic
    short_squeeze_potential: float   # 0-1, squeeze probability


class InstitutionalFlowAnalyzer:
    """
    Analyzes price/volume data to infer institutional activity

    Key insight: Institutions can't hide their footprints entirely.
    Large orders affect volume patterns, price action, and leave
    statistical signatures we can detect.
    """

    def __init__(self):
        pass

    def calculate_unusual_volume(
        self,
        volume: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        Detect unusual volume spikes that may indicate institutional activity

        High volume with small price moves = accumulation/distribution
        High volume with large price moves = momentum (retail + institutions)
        """
        avg_volume = volume.rolling(lookback).mean()
        std_volume = volume.rolling(lookback).std()

        # Z-score of volume
        volume_zscore = (volume - avg_volume) / (std_volume + 1e-8)

        # Normalize to 0-1 range using sigmoid
        unusual_score = 1 / (1 + np.exp(-volume_zscore))

        return unusual_score

    def detect_block_trades(
        self,
        df: pd.DataFrame,
        threshold_multiplier: float = 3.0
    ) -> pd.Series:
        """
        Detect potential block trades (large institutional orders)

        Block trades are typically:
        - 10,000+ shares or $200,000+ value
        - Often executed with minimal price impact
        - May show up as volume spikes without proportional price moves
        """
        # Calculate typical trade size proxy
        avg_volume = df['volume'].rolling(20).mean()
        volume_ratio = df['volume'] / (avg_volume + 1)

        # Price range as percentage
        price_range = (df['high'] - df['low']) / df['close']
        avg_range = price_range.rolling(20).mean()

        # Block trade indicator: high volume but normal/low price range
        # This suggests large orders were absorbed without moving price much
        range_ratio = price_range / (avg_range + 1e-8)

        # High volume + low range = likely block trades
        block_score = volume_ratio / (range_ratio + 1)
        block_score = block_score.clip(0, 10) / 10  # Normalize to 0-1

        return block_score

    def calculate_accumulation_distribution(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Enhanced Accumulation/Distribution indicator

        Classic AD uses close location value (CLV):
        CLV = ((Close - Low) - (High - Close)) / (High - Low)
        AD = Previous AD + Volume * CLV

        Our enhancement: weight by unusual volume
        """
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']

        # Close Location Value
        clv = ((close - low) - (high - close)) / (high - low + 1e-8)
        clv = clv.clip(-1, 1)

        # Money Flow Multiplier
        mf_multiplier = clv * volume

        # Cumulative AD
        ad = mf_multiplier.cumsum()

        # Normalize to recent range
        ad_normalized = (ad - ad.rolling(50).min()) / (ad.rolling(50).max() - ad.rolling(50).min() + 1e-8)

        # Rate of change (is AD increasing or decreasing?)
        ad_roc = ad.diff(5) / (ad.rolling(20).std() + 1e-8)

        return ad_roc.clip(-3, 3) / 3  # Normalize to -1 to 1

    def calculate_money_flow_index(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Money Flow Index (MFI) - Volume-weighted RSI

        High MFI (>80) with falling price = accumulation
        Low MFI (<20) with rising price = distribution
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        # Positive and negative money flow
        price_diff = typical_price.diff()
        positive_flow = money_flow.where(price_diff > 0, 0)
        negative_flow = money_flow.where(price_diff < 0, 0)

        # Rolling sums
        positive_sum = positive_flow.rolling(period).sum()
        negative_sum = negative_flow.rolling(period).sum()

        # Money Flow Ratio
        mfr = positive_sum / (negative_sum + 1e-8)

        # MFI
        mfi = 100 - (100 / (1 + mfr))

        return mfi / 100  # Normalize to 0-1

    def calculate_smart_money_index(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Smart Money Index (SMI)

        Theory: Institutions trade in the last hour, retail trades at open
        Open = Retail sentiment
        Close = Institutional sentiment

        SMI = Previous SMI + (Close - Open)
        """
        daily_change = df['close'] - df['open']

        # Cumulative SMI
        smi = daily_change.cumsum()

        # Normalize by recent volatility
        smi_normalized = smi.diff(5) / (df['close'].rolling(20).std() + 1e-8)

        return smi_normalized.clip(-3, 3) / 3

    def calculate_vwap_divergence(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        VWAP Divergence

        Price above VWAP = buyers in control (accumulation)
        Price below VWAP = sellers in control (distribution)

        Divergence from VWAP can indicate institutional positioning
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        # Distance from VWAP as percentage
        vwap_divergence = (df['close'] - vwap) / vwap

        return vwap_divergence.clip(-0.1, 0.1) * 10  # Normalize to -1 to 1

    def calculate_obv_trend(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        On-Balance Volume (OBV) trend analysis

        Rising OBV with flat price = accumulation
        Falling OBV with flat price = distribution
        """
        price_change = df['close'].diff()

        # OBV calculation
        obv = np.where(
            price_change > 0, df['volume'],
            np.where(price_change < 0, -df['volume'], 0)
        )
        obv = pd.Series(obv, index=df.index).cumsum()

        # OBV trend (rate of change)
        obv_ma = obv.rolling(20).mean()
        obv_trend = (obv - obv_ma) / (obv.rolling(50).std() + 1e-8)

        return obv_trend.clip(-3, 3) / 3

    def detect_price_volume_divergence(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Detect divergences between price and volume trends

        Rising price + falling volume = weak rally (distribution)
        Falling price + rising volume = capitulation (potential bottom)
        Falling price + falling volume = weak decline (accumulation)
        """
        price_trend = df['close'].pct_change(10)  # 10-day price trend
        volume_trend = df['volume'].pct_change(10)  # 10-day volume trend

        # Divergence score
        # Positive = bullish divergence (institutions accumulating)
        # Negative = bearish divergence (institutions distributing)
        divergence = -price_trend * volume_trend  # Opposite signs = divergence

        return divergence.clip(-0.5, 0.5) * 2

    def calculate_weak_hands_score(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Weak Hands Score

        Detects when retail investors are panic selling
        (opportunity for institutional accumulation)

        Signals:
        - High volume on down days
        - Price below key moving averages
        - Increasing volatility
        - Gap downs
        """
        # Price below 50 SMA
        below_50sma = (df['close'] < df['close'].rolling(50).mean()).astype(float)

        # Price below 200 SMA
        below_200sma = (df['close'] < df['close'].rolling(200).mean()).astype(float)

        # High volume on down days
        down_day = (df['close'] < df['open']).astype(float)
        high_volume = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)
        panic_volume = down_day * high_volume

        # Increasing volatility
        volatility = df['close'].pct_change().rolling(10).std()
        vol_increasing = (volatility > volatility.rolling(20).mean()).astype(float)

        # Gap downs
        gap_down = ((df['open'] < df['close'].shift(1) * 0.98)).astype(float)

        # Combine signals
        weak_hands = (
            0.2 * below_50sma +
            0.2 * below_200sma +
            0.3 * panic_volume +
            0.2 * vol_increasing +
            0.1 * gap_down
        )

        return weak_hands

    def calculate_short_squeeze_potential(
        self,
        df: pd.DataFrame,
        short_interest_ratio: float = None
    ) -> pd.Series:
        """
        Short Squeeze Potential Score

        High score when:
        - Price compressed near support
        - Volume declining (shorts complacent)
        - Sudden volume spike with price rise
        - Low float / high short interest (if available)
        """
        # Price near recent lows
        recent_low = df['low'].rolling(20).min()
        near_low = 1 - (df['close'] - recent_low) / (recent_low + 1e-8)
        near_low = near_low.clip(0, 1)

        # Volume compression then expansion
        vol_ma = df['volume'].rolling(20).mean()
        vol_compressed = df['volume'] < vol_ma * 0.7
        vol_expanding = df['volume'] > vol_ma * 1.5

        # Setup: compressed volume followed by expansion with price rise
        price_rising = df['close'] > df['close'].shift(1)
        squeeze_signal = vol_expanding & price_rising

        # Base score
        squeeze_potential = (
            0.4 * near_low +
            0.3 * vol_compressed.astype(float).rolling(5).mean() +
            0.3 * squeeze_signal.astype(float)
        )

        # Boost if short interest data available
        if short_interest_ratio and short_interest_ratio > 20:
            squeeze_potential *= (1 + short_interest_ratio / 100)
            squeeze_potential = squeeze_potential.clip(0, 1)

        return squeeze_potential

    def analyze(
        self,
        df: pd.DataFrame,
        short_interest: float = None
    ) -> pd.DataFrame:
        """
        Run full institutional flow analysis

        Returns DataFrame with all calculated features
        """
        features = pd.DataFrame(index=df.index)

        # Volume analysis
        features['unusual_volume'] = self.calculate_unusual_volume(df['volume'])
        features['block_trade_prob'] = self.detect_block_trades(df)

        # Flow indicators
        features['accumulation_distribution'] = self.calculate_accumulation_distribution(df)
        features['money_flow_index'] = self.calculate_money_flow_index(df)
        features['smart_money_index'] = self.calculate_smart_money_index(df)
        features['vwap_divergence'] = self.calculate_vwap_divergence(df)
        features['obv_trend'] = self.calculate_obv_trend(df)
        features['pv_divergence'] = self.detect_price_volume_divergence(df)

        # Sentiment indicators
        features['weak_hands_score'] = self.calculate_weak_hands_score(df)
        features['squeeze_potential'] = self.calculate_short_squeeze_potential(df, short_interest)

        # Composite scores
        # Smart money (institutional) direction
        features['smart_money_flow'] = (
            0.25 * features['accumulation_distribution'] +
            0.25 * features['smart_money_index'] +
            0.25 * features['obv_trend'] +
            0.25 * features['vwap_divergence']
        )

        # Retail direction (inverse of smart money when weak hands high)
        features['retail_flow'] = -features['smart_money_flow'] * features['weak_hands_score']

        # Fill NaN values
        features = features.fillna(0)

        return features

    def get_signals(
        self,
        df: pd.DataFrame,
        short_interest: float = None
    ) -> InstitutionalSignals:
        """
        Get latest institutional signals as a summary object
        """
        features = self.analyze(df, short_interest)

        if len(features) == 0:
            return InstitutionalSignals(0, 0, 0, 0, 0, 0, 0, 0)

        latest = features.iloc[-1]

        return InstitutionalSignals(
            unusual_volume_score=float(latest['unusual_volume']),
            block_trade_indicator=float(latest['block_trade_prob']),
            accumulation_score=float(latest['accumulation_distribution']),
            distribution_score=float(-latest['accumulation_distribution']),
            smart_money_flow=float(latest['smart_money_flow']),
            retail_flow=float(latest['retail_flow']),
            weak_hands_score=float(latest['weak_hands_score']),
            short_squeeze_potential=float(latest['squeeze_potential'])
        )


def get_short_interest(symbol: str) -> Optional[float]:
    """
    Attempt to get short interest data

    Note: Real-time short interest requires paid data.
    This uses Yahoo Finance which has limited/delayed data.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        shares_short = info.get('sharesShort', 0) or 0
        float_shares = info.get('floatShares', 0) or 0

        if float_shares > 0:
            return (shares_short / float_shares) * 100  # As percentage

        shares_outstanding = info.get('sharesOutstanding', 0) or 0
        if shares_outstanding > 0:
            return (shares_short / shares_outstanding) * 100

    except Exception as e:
        logger.debug(f"Could not get short interest for {symbol}: {e}")

    return None


def analyze_symbol(
    symbol: str,
    days: int = 252
) -> Tuple[pd.DataFrame, InstitutionalSignals]:
    """
    Full institutional flow analysis for a symbol
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d")

        if df.empty:
            return pd.DataFrame(), None

        # Clean columns
        df.columns = [c.lower() for c in df.columns]

        # Get short interest if available
        short_interest = get_short_interest(symbol)

        # Run analysis
        analyzer = InstitutionalFlowAnalyzer()
        features = analyzer.analyze(df, short_interest)
        signals = analyzer.get_signals(df, short_interest)

        # Merge features with price data
        result = pd.concat([df, features], axis=1)

        return result, signals

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return pd.DataFrame(), None


if __name__ == "__main__":
    # Test analysis
    test_symbols = ["NVDA", "AAPL", "GME", "AMC", "TSLA"]

    print("Institutional Flow Analysis Test\n")
    print("=" * 70)

    for symbol in test_symbols:
        df, signals = analyze_symbol(symbol, days=60)

        if signals:
            print(f"\n{symbol}:")
            print(f"  Unusual Volume:     {signals.unusual_volume_score:.2f}")
            print(f"  Block Trade Prob:   {signals.block_trade_indicator:.2f}")
            print(f"  Smart Money Flow:   {signals.smart_money_flow:+.2f}")
            print(f"  Retail Flow:        {signals.retail_flow:+.2f}")
            print(f"  Weak Hands Score:   {signals.weak_hands_score:.2f}")
            print(f"  Squeeze Potential:  {signals.short_squeeze_potential:.2f}")

            # Interpretation
            if signals.smart_money_flow > 0.3 and signals.weak_hands_score > 0.5:
                print(f"  → SIGNAL: Institutions accumulating while retail panics")
            elif signals.smart_money_flow < -0.3 and signals.weak_hands_score < 0.3:
                print(f"  → SIGNAL: Institutions distributing, retail euphoric")
            elif signals.short_squeeze_potential > 0.6:
                print(f"  → SIGNAL: Short squeeze setup detected")
