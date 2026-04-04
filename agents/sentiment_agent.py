"""
Sentiment Agent — live sentiment analysis for the Trade Advisor.

Uses only:
  - NewsAPI (or simulated headlines) for current news
  - FinBERT (or TextBlob fallback) to score each headline

FinSen historical data is NOT used here — it is only used in backtesting
via backtesting/sentiment_cache.py (pre-processed parquet file).
"""
import logging
from typing import Dict

from analysis.sentiment_analyzer import SentimentAnalyzer
from data.news_fetcher import NewsFetcher

logger = logging.getLogger(__name__)


class SentimentAgent:
    """Generates a live sentiment signal from current news headlines."""

    def __init__(self, sentiment_analyzer: SentimentAnalyzer) -> None:
        self._analyzer = sentiment_analyzer
        self._news_fetcher = NewsFetcher()

    def analyze(self, ticker: str) -> Dict:
        """
        Return live sentiment signal dict:
        {score, trend, news_count, historical_context, live_headlines, scored_headlines}
        """
        articles = self._news_fetcher.fetch_news(ticker, days_back=7)
        headlines = [a.get("title", "") for a in articles if a.get("title")]

        scored = self._analyzer.analyze_news_batch(headlines)
        score = self._analyzer.get_aggregate_sentiment(headlines)
        news_count = len(headlines)

        # Derive simple trend from score magnitude
        if score > 0.15:
            trend = "improving"
        elif score < -0.15:
            trend = "deteriorating"
        else:
            trend = "stable"

        return {
            "score": round(score, 4),
            "trend": trend,
            "news_count": news_count,
            "historical_context": "Live news only (FinSen data used in backtesting)",
            "live_headlines": headlines[:5],
            "scored_headlines": scored[:8],
        }
