"""
Technical Indicators — RSI, SMA, and derived trading signals.
All calculations work on plain Python lists or pandas Series.
"""
import logging
from typing import List, Union

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

_Numeric = Union[List[float], pd.Series, np.ndarray]


class TechnicalAnalyzer:
    """Computes technical indicators and generates trading signals."""

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def calculate_rsi(
        self, prices: _Numeric, period: int = config.RSI_PERIOD
    ) -> float:
        """
        Compute Relative Strength Index for the last value in *prices*.
        Returns a float in [0, 100], or 50.0 if there is insufficient data.
        """
        s = pd.Series(prices, dtype=float).dropna()
        if len(s) < period + 1:
            logger.debug("Not enough data for RSI (need %d+1, got %d).", period, len(s))
            return 50.0
        delta = s.diff().dropna()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        # When loss == 0: all gains → RSI = 100; when gain == 0: all losses → RSI = 0
        last_gain = float(gain.iloc[-1]) if not gain.empty else 0.0
        last_loss = float(loss.iloc[-1]) if not loss.empty else 0.0
        if last_loss == 0:
            return 100.0 if last_gain > 0 else 50.0
        rs = last_gain / last_loss
        return round(100 - (100 / (1 + rs)), 4)

    def calculate_sma(self, prices: _Numeric, period: int) -> float:
        """
        Return the Simple Moving Average of the last *period* values in *prices*.
        Returns the last price if there is insufficient data.
        """
        s = pd.Series(prices, dtype=float).dropna()
        if len(s) < period:
            return float(s.iloc[-1]) if not s.empty else 0.0
        return float(s.rolling(period).mean().iloc[-1])

    def calculate_sma_series(self, prices: _Numeric, period: int) -> pd.Series:
        """Return the full SMA series (for charting)."""
        return pd.Series(prices, dtype=float).rolling(period).mean()

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        rsi: float,
        sma_short: float,
        sma_long: float,
        current_price: float = 0.0,
    ) -> str:
        """
        Generate a trading signal string: 'BUY', 'SELL', or 'NEUTRAL'.

        Rules:
        - BUY  : RSI < oversold threshold AND short SMA > long SMA (golden cross)
        - SELL : RSI > overbought threshold AND short SMA < long SMA (death cross)
        - NEUTRAL: everything else
        """
        oversold = config.RSI_OVERSOLD
        overbought = config.RSI_OVERBOUGHT

        bullish_momentum = sma_short > sma_long
        bearish_momentum = sma_short < sma_long

        if rsi < oversold and bullish_momentum:
            return "BUY"
        if rsi > overbought and bearish_momentum:
            return "SELL"
        # Softer signals
        if rsi < oversold:
            return "BUY"
        if rsi > overbought:
            return "SELL"
        return "NEUTRAL"

    def get_technical_score(
        self,
        rsi: float,
        current_price: float,
        sma_short: float,
        sma_long: float,
    ) -> float:
        """
        Return a normalised score in [-1, 1].
        +1 = very bullish, -1 = very bearish.
        """
        # RSI component: map [0,100] → [-1,1], centred at 50
        rsi_score = (50 - rsi) / 50  # oversold (+) overbought (-)
        rsi_score = max(-1.0, min(1.0, rsi_score))

        # SMA component
        if sma_long == 0:
            sma_score = 0.0
        else:
            spread = (sma_short - sma_long) / sma_long
            sma_score = max(-1.0, min(1.0, spread * 10))

        # Price vs SMA_short
        if sma_short == 0:
            price_score = 0.0
        else:
            price_spread = (current_price - sma_short) / sma_short
            price_score = max(-1.0, min(1.0, price_spread * 5))

        return round((rsi_score * 0.5 + sma_score * 0.3 + price_score * 0.2), 4)

    def get_sma_trend(self, sma_short: float, sma_long: float) -> str:
        """Return 'bullish', 'bearish', or 'neutral' based on SMA crossover."""
        if sma_short > sma_long * 1.005:
            return "bullish"
        if sma_short < sma_long * 0.995:
            return "bearish"
        return "neutral"
