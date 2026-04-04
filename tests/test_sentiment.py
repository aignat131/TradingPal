"""Unit tests for SentimentAnalyzer (uses TextBlob fallback — no GPU required)."""
import pytest
from unittest.mock import patch, MagicMock
from analysis.sentiment_analyzer import SentimentAnalyzer


@pytest.fixture
def analyzer():
    """Create analyzer with FinBERT forcibly disabled to use TextBlob fallback."""
    with patch("analysis.sentiment_analyzer.SentimentAnalyzer._load_model"):
        a = SentimentAnalyzer()
        a._use_finbert = False
        a._pipeline = None
    return a


def test_analyze_empty_text(analyzer):
    result = analyzer.analyze_text("")
    assert result["label"] == "neutral"
    assert result["score"] == 0.0


def test_analyze_positive_text(analyzer):
    result = analyzer.analyze_text("The company reported record profits and strong growth.")
    assert result["label"] in ("positive", "neutral", "negative")
    assert -1.0 <= result["score"] <= 1.0


def test_analyze_negative_text(analyzer):
    result = analyzer.analyze_text("The company faces bankruptcy and massive losses.")
    assert result["label"] in ("negative", "neutral")


def test_analyze_news_batch_returns_list(analyzer):
    news = [
        "Company beats earnings estimates",
        "Stock drops on weak guidance",
        "Market remains volatile",
    ]
    results = analyzer.analyze_news_batch(news)
    assert len(results) == 3
    for r in results:
        assert "label" in r
        assert "score" in r
        assert "text" in r


def test_analyze_batch_with_dicts(analyzer):
    news = [{"title": "Strong revenue growth"}, {"title": "Losses mount"}]
    results = analyzer.analyze_news_batch(news)
    assert len(results) == 2


def test_aggregate_sentiment_range(analyzer):
    news = ["Revenue up 20%", "Supply chain issues persist", "Analysts bullish on outlook"]
    score = analyzer.get_aggregate_sentiment(news)
    assert -1.0 <= score <= 1.0


def test_aggregate_empty_list(analyzer):
    score = analyzer.get_aggregate_sentiment([])
    assert score == 0.0
