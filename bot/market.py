"""
Market snapshot for the daily brief.

Downloads ~6 months of daily prices for every asset in one yfinance call and
reuses TechnicalAnalyzer (RSI / SMA) to score each one.

Unlike data.market_data.MarketData this module NEVER falls back to sample or
random prices: an asset without real data is reported as unavailable.
"""
import logging
from typing import Dict, Iterable, List, Optional

import pandas as pd

import config
from analysis.technical_indicators import TechnicalAnalyzer

logger = logging.getLogger(__name__)

BENCHMARKS = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^VIX": "VIX"}


def _download(symbols: List[str], period: str = "6mo") -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        symbols,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )


def _close_series(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Extract a clean Close series for *symbol* from a (possibly MultiIndex) frame."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if symbol not in df.columns.get_level_values(0):
                return pd.Series(dtype=float)
            close = df[symbol]["Close"]
        else:
            close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna().astype(float)
    except (KeyError, IndexError):
        return pd.Series(dtype=float)


def _pct(new: float, old: float) -> Optional[float]:
    if not old:
        return None
    return round((new - old) / old * 100, 2)


def analyze_series(close: pd.Series, analyzer: Optional[TechnicalAnalyzer] = None) -> Optional[Dict]:
    """Price moves + technical signal for one close series (None if too short)."""
    analyzer = analyzer or TechnicalAnalyzer()
    prices = close.tolist()
    if len(prices) < 2:
        return None
    price = prices[-1]
    result = {
        "price": round(price, 4),
        "change_1d": _pct(price, prices[-2]),
        "change_5d": _pct(price, prices[-6]) if len(prices) >= 6 else None,
        "change_1m": _pct(price, prices[-22]) if len(prices) >= 22 else None,
        "last_date": close.index[-1].strftime("%Y-%m-%d") if hasattr(close.index[-1], "strftime") else "",
        "signal": "NEUTRAL",
        "score": 0.0,
        "rsi": None,
        "sma_trend": "neutral",
        "volatility": None,
    }
    if len(prices) >= config.SMA_LONG + 5:
        rsi = analyzer.calculate_rsi(prices)
        sma_s = analyzer.calculate_sma(prices, config.SMA_SHORT)
        sma_l = analyzer.calculate_sma(prices, config.SMA_LONG)
        returns = close.pct_change().dropna().tail(30)
        result.update({
            "signal": analyzer.generate_signal(rsi, sma_s, sma_l, price),
            "score": analyzer.get_technical_score(rsi, price, sma_s, sma_l),
            "rsi": round(rsi, 1),
            "sma_trend": analyzer.get_sma_trend(sma_s, sma_l),
            "volatility": round(float(returns.std() * (252 ** 0.5) * 100), 1) if len(returns) > 5 else None,
        })
    return result


def get_snapshot(symbols: Iterable[str]) -> Dict[str, Optional[Dict]]:
    """Return {symbol: analysis dict or None} for watchlist + benchmark symbols."""
    symbols = list(dict.fromkeys(list(symbols) + list(BENCHMARKS)))
    try:
        df = _download(symbols)
    except Exception as exc:
        logger.error("yfinance batch download failed: %s", exc)
        return {s: None for s in symbols}

    analyzer = TechnicalAnalyzer()
    snapshot = {}
    for sym in symbols:
        snapshot[sym] = analyze_series(_close_series(df, sym), analyzer)
        if snapshot[sym] is None:
            logger.warning("No price data for %s.", sym)
    return snapshot


def symbol_exists(symbol: str) -> bool:
    """True if Yahoo Finance has recent prices for *symbol* (used before adding)."""
    try:
        df = _download([symbol], period="1mo")
        return not _close_series(df, symbol).empty
    except Exception as exc:
        logger.warning("Symbol check failed for %s: %s", symbol, exc)
        return False
