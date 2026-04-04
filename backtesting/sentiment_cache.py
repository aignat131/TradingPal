"""
Sentiment Cache — loads the pre-processed FinSen parquet file and provides
fast daily sentiment lookups for the backtest engine.

The parquet is produced by:  python utils/preprocess_finsen.py

Schema:
  date           datetime64
  ticker         str
  avg_score      float   (TextBlob polarity, -1 to +1)
  article_count  int
  positive_count int
  negative_count int
  neutral_count  int
"""
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.path.join("data_cache", "finsen_daily_sentiment.parquet")


class SentimentCache:
    """Fast lookup for pre-processed daily sentiment scores."""

    def __init__(self, parquet_path: str = _DEFAULT_PATH) -> None:
        self._path = parquet_path
        self._df: Optional[pd.DataFrame] = None
        self._available = False
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    def get_score(self, ticker: str, date) -> float:
        """
        Return the average sentiment score for *ticker* on *date*.
        If no exact match, returns the average of the nearest ±7 days.
        Returns 0.0 if unavailable.
        """
        if not self._available:
            return 0.0
        ts = pd.Timestamp(date).normalize()
        tkr = ticker.upper()
        tkr_df = self._df[self._df["ticker"] == tkr]
        if tkr_df.empty:
            return 0.0
        # Exact match
        exact = tkr_df[tkr_df["date"] == ts]
        if not exact.empty:
            return float(exact["avg_score"].iloc[0])
        # Nearest week window
        window = tkr_df[
            (tkr_df["date"] >= ts - pd.Timedelta(days=7))
            & (tkr_df["date"] <= ts + pd.Timedelta(days=7))
        ]
        if window.empty:
            return 0.0
        return round(float(window["avg_score"].mean()), 4)

    def get_window_score(self, ticker: str, start, end) -> float:
        """Return the average sentiment score for *ticker* in a date window."""
        if not self._available:
            return 0.0
        tkr = ticker.upper()
        mask = (
            (self._df["ticker"] == tkr)
            & (self._df["date"] >= pd.Timestamp(start).normalize())
            & (self._df["date"] <= pd.Timestamp(end).normalize())
        )
        sub = self._df[mask]
        if sub.empty:
            return 0.0
        return round(float(sub["avg_score"].mean()), 4)

    def get_available_tickers(self):
        """Return sorted list of tickers in the cache."""
        if not self._available:
            return []
        return sorted(self._df["ticker"].unique().tolist())

    def get_date_range(self):
        """Return (min_date, max_date) of the cache."""
        if not self._available:
            return None, None
        return self._df["date"].min(), self._df["date"].max()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.isfile(self._path):
            logger.warning(
                "Sentiment cache not found at '%s'. "
                "Run: python utils/preprocess_finsen.py",
                self._path,
            )
            return
        try:
            self._df = pd.read_parquet(self._path)
            self._df["date"] = pd.to_datetime(self._df["date"]).dt.normalize()
            self._available = True
            logger.info(
                "Sentiment cache loaded: %d records, %d tickers, %s → %s",
                len(self._df),
                self._df["ticker"].nunique(),
                self._df["date"].min().date(),
                self._df["date"].max().date(),
            )
        except Exception as exc:
            logger.error("Failed to load sentiment cache: %s", exc)
