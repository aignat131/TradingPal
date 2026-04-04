"""
Market data module — fetches live and historical prices via yfinance.
Falls back to sample CSV data when yfinance is unavailable or returns nothing.
"""
import logging
import os
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "sample_prices.csv")


class MarketData:
    """Provides current and historical price data for stocks and crypto."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_price(self, ticker: str) -> Dict:
        """
        Return a dict with current market snapshot:
        {price, change_pct, high_52w, low_52w, volume}
        """
        data = self._fetch_yfinance_snapshot(ticker)
        if data:
            return data
        return self._fallback_snapshot(ticker)

    def get_historical_prices(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Return OHLCV DataFrame for *ticker*.
        *period*: yfinance period string, e.g. '1y', '6mo', '3mo'.
        Returns empty DataFrame on failure.
        """
        try:
            import yfinance as yf

            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as exc:
            logger.warning("yfinance historical failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    def get_historical_prices_range(
        self, ticker: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame for a specific date range (YYYY-MM-DD strings)."""
        try:
            import yfinance as yf

            df = yf.download(
                ticker, start=start, end=end, interval=interval, progress=False
            )
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as exc:
            logger.warning("yfinance range failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    def get_company_info(self, ticker: str) -> Dict:
        """Return basic company info dict from yfinance."""
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info
            return {
                "name": info.get("longName", ticker),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "description": info.get("longBusinessSummary", ""),
            }
        except Exception as exc:
            logger.warning("Could not fetch company info for %s: %s", ticker, exc)
            return {"name": ticker, "sector": "N/A", "industry": "N/A",
                    "market_cap": 0, "description": ""}

    def calculate_volatility(self, ticker: str, days: int = 30) -> float:
        """Return annualised historical volatility (%) for the last *days* trading days."""
        try:
            df = self.get_historical_prices(ticker, period=f"{days * 2}d")
            if df.empty:
                return 0.0
            # Use 'Close' column, handle MultiIndex from yfinance
            close = df["Close"]
            if hasattr(close, "squeeze"):
                close = close.squeeze()
            returns = close.pct_change().dropna().tail(days)
            if len(returns) < 5:
                return 0.0
            return float(returns.std() * (252 ** 0.5) * 100)
        except Exception as exc:
            logger.warning("Volatility calc failed for %s: %s", ticker, exc)
            return 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_yfinance_snapshot(self, ticker: str) -> Optional[Dict]:
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            info = t.info
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
            if not price:
                return None
            prev_close = info.get("previousClose") or price
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
            return {
                "price": float(price),
                "change_pct": round(float(change_pct), 2),
                "high_52w": float(info.get("fiftyTwoWeekHigh", price * 1.2)),
                "low_52w": float(info.get("fiftyTwoWeekLow", price * 0.8)),
                "volume": info.get("volume") or info.get("averageVolume", "N/A"),
            }
        except Exception as exc:
            logger.warning("yfinance snapshot failed for %s: %s", ticker, exc)
            return None

    def _fallback_snapshot(self, ticker: str) -> Dict:
        """Look up ticker in sample CSV; generate plausible mock data if not found."""
        try:
            df = pd.read_csv(_SAMPLE_CSV)
            row = df[df["ticker"].str.upper() == ticker.upper()]
            if not row.empty:
                r = row.iloc[0]
                return {
                    "price": float(r.get("price", 100)),
                    "change_pct": float(r.get("change_pct", 0)),
                    "high_52w": float(r.get("high_52w", r.get("price", 100) * 1.3)),
                    "low_52w": float(r.get("low_52w", r.get("price", 100) * 0.7)),
                    "volume": r.get("volume", "N/A"),
                }
        except Exception:
            pass
        import random

        random.seed(hash(ticker) % 9999)
        price = round(random.uniform(10, 500), 2)
        return {
            "price": price,
            "change_pct": round(random.uniform(-5, 5), 2),
            "high_52w": round(price * random.uniform(1.1, 1.5), 2),
            "low_52w": round(price * random.uniform(0.5, 0.9), 2),
            "volume": random.randint(1_000_000, 50_000_000),
        }
