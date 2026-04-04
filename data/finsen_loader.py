"""
FinSen Dataset Loader.

Supports the actual files from the Kaggle dataset:
  https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests

The loader auto-detects whichever CSV is present in the configured directory:

  Priority 1 — raw_partner_headlines.csv
    Columns: headline, url, publisher, date, stock
    ~1M rows of news headlines linked to stock tickers.

  Priority 2 — analyst_ratings_processed.csv
    Columns: title, published_utc (or date), ticker, etc.

  Priority 3 — raw_analyst_ratings.csv
    Columns: title, date, url, publisher, stock

All files are normalised internally to:
  date | ticker | headline | sentiment | sentiment_score

Sentiment scores are computed via TextBlob when not present in the file
(fast, runs at load time on a sample; full scoring is done lazily).
"""
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)

# Files to try, in priority order, relative to the configured directory
_CANDIDATE_FILES = [
    "raw_partner_headlines.csv",
    "analyst_ratings_processed.csv",
    "raw_analyst_ratings.csv",
    "US_Financial_News.csv",  # legacy name kept for compatibility
]

# Column name mappings: each entry maps source columns → internal names
# Format: (date_col, ticker_col, headline_col)
_COLUMN_MAPS = {
    "raw_partner_headlines.csv":       ("date", "stock",  "headline"),
    "analyst_ratings_processed.csv":   ("date", "ticker", "title"),
    "raw_analyst_ratings.csv":         ("date", "stock",  "title"),
    "US_Financial_News.csv":           ("date", "ticker", "headline"),
}


class FinSenLoader:
    """Loads and queries the FinSen historical sentiment dataset."""

    def __init__(self, csv_path: Optional[str] = None) -> None:
        # csv_path can be a direct file path OR a directory containing the CSVs
        self._path = csv_path or config.FINSEN_DATA_PATH
        self._df: Optional[pd.DataFrame] = None
        self._available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(self) -> bool:
        """
        Auto-detect and load whichever supported CSV is present.
        Returns True if successful.
        """
        csv_file = self._resolve_path()
        if csv_file is None:
            logger.warning(
                "No FinSen CSV found in '%s'. Using simulated sentiment. "
                "Expected one of: %s",
                self._path,
                ", ".join(_CANDIDATE_FILES),
            )
            self._available = False
            return False

        try:
            filename = os.path.basename(csv_file)
            date_col, ticker_col, headline_col = _COLUMN_MAPS.get(
                filename, ("date", "ticker", "headline")
            )

            logger.info("Loading FinSen file: %s", csv_file)
            raw = pd.read_csv(csv_file, low_memory=False)
            logger.info("Loaded %d rows. Columns: %s", len(raw), list(raw.columns))

            # --- Normalise columns ---
            df = pd.DataFrame()

            # Date
            if date_col in raw.columns:
                df["date"] = pd.to_datetime(raw[date_col], errors="coerce", utc=False)
            else:
                # Try common alternatives
                for alt in ("published_utc", "publishedAt", "publish_date"):
                    if alt in raw.columns:
                        df["date"] = pd.to_datetime(raw[alt], errors="coerce", utc=False)
                        break
                else:
                    logger.error("No date column found in %s.", filename)
                    return False

            # Strip timezone so comparisons work cleanly
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)

            # Ticker
            if ticker_col in raw.columns:
                df["ticker"] = raw[ticker_col].astype(str).str.upper().str.strip()
            else:
                logger.error("No ticker column found in %s.", filename)
                return False

            # Headline
            if headline_col in raw.columns:
                df["headline"] = raw[headline_col].astype(str)
            else:
                df["headline"] = ""

            # Sentiment score — compute via TextBlob if not present
            if "sentiment_score" in raw.columns:
                df["sentiment_score"] = pd.to_numeric(
                    raw["sentiment_score"], errors="coerce"
                ).fillna(0.0)
            elif "sentiment" in raw.columns and raw["sentiment"].dtype != object:
                df["sentiment_score"] = pd.to_numeric(
                    raw["sentiment"], errors="coerce"
                ).fillna(0.0)
            else:
                logger.info(
                    "No pre-computed sentiment scores — computing via TextBlob "
                    "(this may take a moment)..."
                )
                df["sentiment_score"] = self._score_with_textblob(df["headline"])

            # Sentiment label
            df["sentiment"] = df["sentiment_score"].apply(
                lambda s: "positive" if s > 0.1 else "negative" if s < -0.1 else "neutral"
            )

            # Drop rows with bad dates or tickers
            df.dropna(subset=["date", "ticker"], inplace=True)
            df = df[df["ticker"].str.len() <= 10]  # filter junk tickers

            self._df = df.reset_index(drop=True)
            self._available = True
            logger.info(
                "FinSen dataset ready: %d rows, %d unique tickers.",
                len(self._df),
                self._df["ticker"].nunique(),
            )
            return True

        except Exception as exc:
            logger.error("Failed to load FinSen CSV '%s': %s", csv_file, exc)
            self._available = False
            return False

    def get_historical_sentiment(
        self, ticker: str, start_date: str, end_date: str
    ) -> List[Dict]:
        """
        Return sentiment records for *ticker* between *start_date* and *end_date*.
        Each record: {date, headline, sentiment, sentiment_score}
        Returns simulated data when the CSV is unavailable.
        """
        if not self._available:
            return self._simulate_sentiment(ticker, start_date, end_date)

        mask = (
            (self._df["ticker"] == ticker.upper())
            & (self._df["date"] >= pd.Timestamp(start_date))
            & (self._df["date"] <= pd.Timestamp(end_date))
        )
        subset = self._df[mask].sort_values("date")
        return subset[["date", "headline", "sentiment", "sentiment_score"]].to_dict(
            orient="records"
        )

    def get_sentiment_trend(self, ticker: str, days: int = 30) -> str:
        """Return 'improving', 'stable', or 'deteriorating'."""
        if not self._available:
            return "stable"

        end = datetime.utcnow()
        mid = end - timedelta(days=days)
        start = mid - timedelta(days=days)

        recent = self._avg_score(ticker, mid.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        prior = self._avg_score(ticker, start.strftime("%Y-%m-%d"), mid.strftime("%Y-%m-%d"))

        if recent > prior + 0.05:
            return "improving"
        if recent < prior - 0.05:
            return "deteriorating"
        return "stable"

    def get_ticker_coverage(self) -> List[str]:
        """Return sorted list of all tickers in the dataset."""
        if not self._available:
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM"]
        return sorted(self._df["ticker"].unique().tolist())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_path(self) -> Optional[str]:
        """
        Return the path to the first available CSV file.
        Accepts either a direct file path or a directory.
        """
        p = self._path

        # Direct file path
        if os.path.isfile(p):
            return p

        # Directory — try each candidate filename
        directory = os.path.dirname(p) if not os.path.isdir(p) else p
        for fname in _CANDIDATE_FILES:
            candidate = os.path.join(directory, fname)
            if os.path.isfile(candidate):
                logger.info("Found FinSen file: %s", candidate)
                return candidate

        return None

    def _avg_score(self, ticker: str, start: str, end: str) -> float:
        records = self.get_historical_sentiment(ticker, start, end)
        if not records:
            return 0.0
        scores = [r.get("sentiment_score", 0.0) for r in records]
        return sum(scores) / len(scores)

    @staticmethod
    def _score_with_textblob(headlines: pd.Series) -> pd.Series:
        """Compute polarity scores via TextBlob (fast, no GPU needed)."""
        try:
            from textblob import TextBlob

            return headlines.apply(
                lambda t: round(TextBlob(str(t)).sentiment.polarity, 4) if t else 0.0
            )
        except Exception:
            return pd.Series([0.0] * len(headlines))

    def _simulate_sentiment(
        self, ticker: str, start_date: str, end_date: str
    ) -> List[Dict]:
        """Generate plausible simulated sentiment when CSV is absent."""
        random.seed(hash(ticker) % 1000)
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        records = []
        current = start
        templates = [
            f"{ticker} reports strong quarterly earnings",
            f"Analysts upgrade {ticker} to buy",
            f"{ticker} faces regulatory scrutiny",
            f"{ticker} announces new product line",
            f"Market volatility impacts {ticker} shares",
        ]
        while current <= end:
            if current.weekday() < 5:
                score = round(random.uniform(-0.3, 0.6), 4)
                records.append({
                    "date": current,
                    "headline": random.choice(templates),
                    "sentiment": "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral",
                    "sentiment_score": score,
                })
            current += timedelta(days=1)
        return records
