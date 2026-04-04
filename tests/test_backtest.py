"""Unit tests for BacktestEngine and AccuracyReporter."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from backtesting.backtest_engine import BacktestEngine
from backtesting.accuracy_reporter import AccuracyReporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_finsen():
    loader = MagicMock()
    loader.get_historical_sentiment.return_value = []
    return loader


@pytest.fixture
def sample_price_df():
    """120 trading days of synthetic OHLCV data."""
    dates = pd.date_range("2023-08-01", periods=120, freq="B")
    np.random.seed(42)
    prices = 150.0 + np.cumsum(np.random.randn(120) * 2)
    prices = np.abs(prices)
    df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.02,
            "Low": prices * 0.97,
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 5_000_000, 120),
        },
        index=dates,
    )
    return df


@pytest.fixture
def engine(mock_finsen):
    return BacktestEngine(finsen_loader=mock_finsen)


# ---------------------------------------------------------------------------
# BacktestEngine tests
# ---------------------------------------------------------------------------

def test_generate_signals_returns_dataframe(engine, sample_price_df):
    signals = engine.generate_signals_for_period(
        sample_price_df, "FAKE", "2023-10-01", "2023-12-31"
    )
    assert isinstance(signals, pd.DataFrame)


def test_signals_have_required_columns(engine, sample_price_df):
    signals = engine.generate_signals_for_period(
        sample_price_df, "FAKE", "2023-10-01", "2023-12-31"
    )
    if not signals.empty:
        required = {"close", "rsi", "composite_signal", "tech_score"}
        assert required.issubset(set(signals.columns))


def test_composite_signal_valid_values(engine, sample_price_df):
    signals = engine.generate_signals_for_period(
        sample_price_df, "FAKE", "2023-10-01", "2023-12-31"
    )
    if not signals.empty:
        valid = {"BUY", "SELL", "NEUTRAL"}
        assert set(signals["composite_signal"].unique()).issubset(valid)


def test_calculate_performance_with_no_trades(engine):
    metrics = engine.calculate_performance(trades=[], equity_curve=[], initial_capital=10000)
    assert metrics["total_return_pct"] == 0.0
    assert metrics["total_trades"] == 0


def test_calculate_performance_with_data(engine):
    equity = [
        {"date": pd.Timestamp("2024-01-01"), "value": 10000},
        {"date": pd.Timestamp("2024-06-01"), "value": 11000},
        {"date": pd.Timestamp("2024-12-31"), "value": 10500},
    ]
    trades = [
        {"entry_date": "2024-01-01", "exit_date": "2024-06-01",
         "entry_price": 150, "exit_price": 165, "shares": 10, "pnl": 150, "return_pct": 10},
    ]
    metrics = engine.calculate_performance(trades, equity, initial_capital=10000)
    assert metrics["total_trades"] == 1
    assert metrics["win_rate_pct"] == 100.0


def test_empty_result_on_no_data(engine):
    with patch.object(engine._market, "get_historical_prices_range", return_value=pd.DataFrame()):
        result = engine.run_backtest("FAKE", "2024-01-01", "2024-12-31")
    assert result["signals"].empty
    assert result["trades"] == []


# ---------------------------------------------------------------------------
# AccuracyReporter tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_signals():
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    close_prices = [100 + i for i in range(20)]
    signals = ["BUY", "NEUTRAL", "SELL"] * 6 + ["BUY", "NEUTRAL"]
    return pd.DataFrame(
        {"close": close_prices, "composite_signal": signals[:20]},
        index=dates,
    )


def test_compare_with_reality_adds_columns(sample_signals):
    reporter = AccuracyReporter()
    result = reporter.compare_with_reality(sample_signals, lookahead_days=3)
    assert "forward_return_pct" in result.columns
    assert "correct" in result.columns


def test_accuracy_metrics_structure(sample_signals):
    reporter = AccuracyReporter()
    augmented = reporter.compare_with_reality(sample_signals, lookahead_days=3)
    metrics = reporter.calculate_accuracy_metrics(augmented)
    assert "overall_accuracy_pct" in metrics
    assert "buy_precision_pct" in metrics
    assert "sell_precision_pct" in metrics


def test_generate_report_returns_csv_string(sample_signals):
    reporter = AccuracyReporter()
    result = {
        "ticker": "TEST",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_capital": 10000,
        "signals": sample_signals,
        "trades": [],
        "metrics": {},
    }
    csv_str, _ = reporter.generate_report(result)
    assert "TradingPal-AI" in csv_str
    assert "TEST" in csv_str
