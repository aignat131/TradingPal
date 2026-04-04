"""
Sentiment Analyzer — uses FinBERT (ProsusAI/finbert) to classify financial text.
Falls back to TextBlob when transformers/torch are unavailable.
"""
import logging
from typing import Dict, List, Union

import config

logger = logging.getLogger(__name__)

_LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


class SentimentAnalyzer:
    """
    Classifies financial news sentiment.
    Uses FinBERT when available, TextBlob otherwise.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._use_finbert = False
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_text(self, text: str) -> Dict:
        """
        Analyse a single text snippet.
        Returns: {label: 'positive'|'negative'|'neutral', score: float [-1,1]}
        """
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.0}
        if self._use_finbert:
            return self._finbert_single(text)
        return self._textblob_single(text)

    def analyze_news_batch(self, news_list: List[Union[str, Dict]]) -> List[Dict]:
        """
        Analyse a list of news items (strings or dicts with 'title' key).
        Returns a list of {text, label, score} dicts.
        """
        results = []
        for item in news_list:
            text = item if isinstance(item, str) else item.get("title", "")
            result = self.analyze_text(text)
            results.append({"text": text, **result})
        return results

    def get_aggregate_sentiment(
        self, news_list: List[Union[str, Dict]]
    ) -> float:
        """
        Return a single aggregate sentiment score in [-1, 1].
        Computed as the weighted mean of individual scores.
        """
        batch = self.analyze_news_batch(news_list)
        if not batch:
            return 0.0
        scores = [r["score"] for r in batch]
        return round(sum(scores) / len(scores), 4)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=config.FINBERT_MODEL,
                top_k=1,
            )
            self._use_finbert = True
            logger.info("FinBERT model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "FinBERT unavailable (%s). Falling back to TextBlob.", exc
            )
            self._use_finbert = False

    def _finbert_single(self, text: str) -> Dict:
        try:
            # Truncate to 512 tokens worth of characters
            truncated = text[:512]
            result = self._pipeline(truncated)
            # result is [[{'label': ..., 'score': ...}]]
            if result and result[0]:
                top = result[0][0] if isinstance(result[0], list) else result[0]
                label = top["label"].lower()
                confidence = float(top["score"])
                score = _LABEL_MAP.get(label, 0.0) * confidence
                return {"label": label, "score": round(score, 4)}
        except Exception as exc:
            logger.warning("FinBERT inference failed: %s", exc)
        return {"label": "neutral", "score": 0.0}

    def _textblob_single(self, text: str) -> Dict:
        try:
            from textblob import TextBlob

            polarity = TextBlob(text).sentiment.polarity  # [-1, 1]
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
            return {"label": label, "score": round(polarity, 4)}
        except Exception as exc:
            logger.warning("TextBlob analysis failed: %s", exc)
            return {"label": "neutral", "score": 0.0}
