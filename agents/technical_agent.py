"""
Technical Agent — wraps TechnicalAnalyzer and MarketData to produce
a standardised technical signal dict.
"""
import logging
from typing import Dict

from analysis.technical_indicators import TechnicalAnalyzer
from data.market_data import MarketData
import config

logger = logging.getLogger(__name__)


class TechnicalAgent:
    """Generates a technical analysis signal for a given ticker."""

    def __init__(self) -> None:
        self._analyzer = TechnicalAnalyzer()
        self._market = MarketData()

    def analyze(self, ticker: str) -> Dict:
        """
        Return technical signal dict:
        {signal, score, rsi, sma_short, sma_long, sma_trend, volatility, prices_available}
        """
        df = self._market.get_historical_prices(ticker, period="6mo")
        if df.empty:
            logger.warning("No price history for %s — returning neutral signal.", ticker)
            return self._neutral(ticker)

        # Handle MultiIndex columns (yfinance quirk)
        close = df["Close"]
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        prices = close.dropna().tolist()

        if len(prices) < config.SMA_LONG + 5:
            return self._neutral(ticker)

        rsi = self._analyzer.calculate_rsi(prices)
        sma_short = self._analyzer.calculate_sma(prices, config.SMA_SHORT)
        sma_long = self._analyzer.calculate_sma(prices, config.SMA_LONG)
        current_price = prices[-1]
        signal = self._analyzer.generate_signal(rsi, sma_short, sma_long, current_price)
        score = self._analyzer.get_technical_score(rsi, current_price, sma_short, sma_long)
        sma_trend = self._analyzer.get_sma_trend(sma_short, sma_long)
        volatility = self._market.calculate_volatility(ticker)

        return {
            "signal": signal,
            "score": score,
            "rsi": round(rsi, 2),
            "sma_short": round(sma_short, 4),
            "sma_long": round(sma_long, 4),
            "sma_trend": sma_trend,
            "volatility": round(volatility, 2),
            "prices_available": len(prices),
        }

    def _neutral(self, ticker: str) -> Dict:
        return {
            "signal": "NEUTRAL",
            "score": 0.0,
            "rsi": 50.0,
            "sma_short": 0.0,
            "sma_long": 0.0,
            "sma_trend": "neutral",
            "volatility": 0.0,
            "prices_available": 0,
        }
