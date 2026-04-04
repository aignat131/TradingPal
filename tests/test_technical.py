"""Unit tests for TechnicalAnalyzer."""
import pytest
from analysis.technical_indicators import TechnicalAnalyzer


@pytest.fixture
def analyzer():
    return TechnicalAnalyzer()


@pytest.fixture
def rising_prices():
    return [100 + i * 0.5 for i in range(60)]


@pytest.fixture
def falling_prices():
    return [200 - i * 0.5 for i in range(60)]


def test_rsi_neutral_with_insufficient_data(analyzer):
    assert analyzer.calculate_rsi([100, 101, 102]) == 50.0


def test_rsi_rising_market_is_high(analyzer, rising_prices):
    rsi = analyzer.calculate_rsi(rising_prices)
    assert rsi > 60, f"Expected RSI > 60 for rising market, got {rsi}"


def test_rsi_falling_market_is_low(analyzer, falling_prices):
    rsi = analyzer.calculate_rsi(falling_prices)
    assert rsi < 40, f"Expected RSI < 40 for falling market, got {rsi}"


def test_rsi_range(analyzer, rising_prices):
    rsi = analyzer.calculate_rsi(rising_prices)
    assert 0 <= rsi <= 100


def test_sma_simple(analyzer):
    prices = [10.0, 20.0, 30.0, 40.0, 50.0]
    sma = analyzer.calculate_sma(prices, period=5)
    assert sma == pytest.approx(30.0)


def test_sma_insufficient_data_returns_last_price(analyzer):
    prices = [100.0, 110.0]
    result = analyzer.calculate_sma(prices, period=10)
    assert result == pytest.approx(110.0)


def test_generate_signal_buy(analyzer):
    # RSI below 30 and short SMA > long SMA → BUY
    signal = analyzer.generate_signal(rsi=25, sma_short=105, sma_long=100)
    assert signal == "BUY"


def test_generate_signal_sell(analyzer):
    # RSI above 70 and short SMA < long SMA → SELL
    signal = analyzer.generate_signal(rsi=75, sma_short=95, sma_long=100)
    assert signal == "SELL"


def test_generate_signal_neutral(analyzer):
    signal = analyzer.generate_signal(rsi=50, sma_short=100, sma_long=100)
    assert signal == "NEUTRAL"


def test_technical_score_bullish(analyzer):
    score = analyzer.get_technical_score(rsi=20, current_price=110, sma_short=105, sma_long=100)
    assert score > 0, f"Expected positive score for bullish conditions, got {score}"


def test_technical_score_bearish(analyzer):
    score = analyzer.get_technical_score(rsi=80, current_price=90, sma_short=95, sma_long=100)
    assert score < 0, f"Expected negative score for bearish conditions, got {score}"


def test_technical_score_range(analyzer):
    score = analyzer.get_technical_score(rsi=50, current_price=100, sma_short=100, sma_long=100)
    assert -1 <= score <= 1
