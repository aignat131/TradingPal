"""
News Fetcher — retrieves recent news headlines for a ticker via NewsAPI.
Falls back to simulated news headlines when the API key is missing or the call fails.
"""
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List

import config

logger = logging.getLogger(__name__)


class NewsFetcher:
    """Fetches news articles relevant to a stock ticker."""

    def __init__(self) -> None:
        self._api_key = config.NEWS_API_KEY
        self._client = None
        if self._api_key:
            try:
                from newsapi import NewsApiClient

                self._client = NewsApiClient(api_key=self._api_key)
                logger.info("NewsAPI client initialised.")
            except Exception as exc:
                logger.warning("Failed to init NewsAPI client: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """
        Return a list of news article dicts for *ticker*.
        Each dict: {title, description, url, published_at, source}
        """
        if self._client:
            articles = self._fetch_from_api(ticker, days_back)
            if articles:
                return articles
        return self._simulate_news(ticker, days_back)

    def fetch_headlines(self, ticker: str, limit: int = 10) -> List[str]:
        """Return a list of plain headline strings (at most *limit*)."""
        articles = self.fetch_news(ticker, days_back=7)
        return [a["title"] for a in articles[:limit] if a.get("title")]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_from_api(self, ticker: str, days_back: int) -> List[Dict]:
        try:
            from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime(
                "%Y-%m-%d"
            )
            response = self._client.get_everything(
                q=ticker,
                from_param=from_date,
                language="en",
                sort_by="relevancy",
                page_size=20,
            )
            articles = response.get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                    "source": a.get("source", {}).get("name", ""),
                }
                for a in articles
            ]
        except Exception as exc:
            logger.warning("NewsAPI call failed: %s", exc)
            return []

    def _simulate_news(self, ticker: str, days_back: int) -> List[Dict]:
        """Generate plausible simulated news when API is unavailable."""
        random.seed(hash(ticker + str(days_back)) % 10_000)
        templates = [
            ("{ticker} beats analyst expectations with strong revenue growth",
             "positive"),
            ("{ticker} announces share buyback program worth $2B",
             "positive"),
            ("{ticker} partners with leading AI company for new platform",
             "positive"),
            ("{ticker} shares dip on broader market selloff",
             "negative"),
            ("{ticker} faces supply chain headwinds in Q4",
             "negative"),
            ("{ticker} CFO comments on macroeconomic uncertainty",
             "neutral"),
            ("Investors eye {ticker} ahead of earnings report",
             "neutral"),
            ("{ticker} upgrades guidance for fiscal year",
             "positive"),
        ]
        articles = []
        for i in range(min(8, days_back * 2)):
            template, _ = random.choice(templates)
            date = datetime.utcnow() - timedelta(days=random.randint(0, days_back))
            articles.append(
                {
                    "title": template.format(ticker=ticker),
                    "description": f"Simulated news article for {ticker}.",
                    "url": "",
                    "published_at": date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "source": "Simulated",
                }
            )
        return articles
