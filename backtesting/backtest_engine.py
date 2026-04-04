"""
Backtest Engine — simulates TradingPal-AI signals over a historical period.

How it works
------------
1. Downloads daily OHLCV data for the full period via yfinance (~252 days for 1 year).
2. For each trading day (rolling window):
   - Calculates RSI(14) and SMA(20/50) on the preceding price history.
   - Looks up pre-processed FinSen sentiment from SentimentCache (parquet).
   - Produces a composite signal: BUY / SELL / NEUTRAL.
3. Simulates trades:
   - BUY signal  → open a long position (position sizing by risk%).
   - SELL signal → close any open position and record P&L.
4. Computes performance metrics:
   Total Return, Annualised Return, Sharpe Ratio, Max Drawdown, Win Rate.

Requires the sentiment cache to be built first:
    python utils/preprocess_finsen.py
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.technical_indicators import TechnicalAnalyzer
from backtesting.sentiment_cache import SentimentCache
from data.market_data import MarketData
import config

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Runs historical signal simulations and computes performance statistics."""

    def __init__(self, sentiment_cache: Optional[SentimentCache] = None) -> None:
        self._market = MarketData()
        self._tech = TechnicalAnalyzer()
        self._cache = sentiment_cache or SentimentCache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        ticker: str,
        start_date: str = config.BACKTEST_START,
        end_date: str = config.BACKTEST_END,
        initial_capital: float = config.INITIAL_CAPITAL,
        risk_percent: float = config.DEFAULT_RISK_PERCENT,
    ) -> Dict:
        """
        Run a full backtest for *ticker* between *start_date* and *end_date*.

        Returns a result dict with:
          signals      — DataFrame of daily signals and prices
          trades       — list of completed trade dicts
          equity_curve — list of (date, portfolio_value) tuples
          metrics      — performance metrics dict
        """
        # Fetch prices: fetch extra history for indicator warm-up
        df = self._market.get_historical_prices_range(
            ticker,
            start=self._offset_date(start_date, -120),
            end=end_date,
        )
        if df.empty:
            logger.warning("No price data for %s in range %s–%s.", ticker, start_date, end_date)
            return self._empty_result(ticker)

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        signals_df = self.generate_signals_for_period(df, ticker, start_date, end_date)
        trades, equity_curve = self._simulate_trades(
            signals_df, initial_capital, risk_percent
        )
        metrics = self.calculate_performance(trades, equity_curve, initial_capital)

        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "signals": signals_df,
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
        }

    def generate_signals_for_period(
        self,
        df: pd.DataFrame,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Generate daily signals for each date in [start_date, end_date].
        Uses a rolling window of prior prices so each signal is realistic
        (no look-ahead bias).

        Returns a DataFrame with columns:
          date, open, high, low, close, volume,
          rsi, sma_short, sma_long, tech_signal, tech_score,
          hist_sentiment, composite_signal
        """
        close = df["Close"].squeeze()
        rows = []
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        for i, date in enumerate(df.index):
            if date < start_ts or date > end_ts:
                continue
            # Use all prices UP TO (but not including) current day — no look-ahead
            hist_prices = close.iloc[:i].tolist()
            if len(hist_prices) < config.SMA_LONG + 5:
                continue

            rsi = self._tech.calculate_rsi(hist_prices)
            sma_s = self._tech.calculate_sma(hist_prices, config.SMA_SHORT)
            sma_l = self._tech.calculate_sma(hist_prices, config.SMA_LONG)
            current_price = float(close.iloc[i])
            tech_signal = self._tech.generate_signal(rsi, sma_s, sma_l, current_price)
            tech_score = self._tech.get_technical_score(rsi, current_price, sma_s, sma_l)

            # Historical sentiment (FinSen or simulated)
            hist_sent = self._get_hist_sentiment(ticker, date)

            # Composite: 60% technical, 40% sentiment
            composite_score = tech_score * 0.6 + hist_sent * 0.4
            if composite_score > 0.15:
                composite_signal = "BUY"
            elif composite_score < -0.15:
                composite_signal = "SELL"
            else:
                composite_signal = "NEUTRAL"

            rows.append(
                {
                    "date": date,
                    "open": float(df["Open"].squeeze().iloc[i]),
                    "high": float(df["High"].squeeze().iloc[i]),
                    "low": float(df["Low"].squeeze().iloc[i]),
                    "close": current_price,
                    "volume": float(df["Volume"].squeeze().iloc[i]),
                    "rsi": round(rsi, 2),
                    "sma_short": round(sma_s, 4),
                    "sma_long": round(sma_l, 4),
                    "tech_signal": tech_signal,
                    "tech_score": round(tech_score, 4),
                    "hist_sentiment": round(hist_sent, 4),
                    "composite_signal": composite_signal,
                    "composite_score": round(composite_score, 4),
                }
            )

        return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

    def calculate_performance(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
        initial_capital: float,
    ) -> Dict:
        """
        Compute performance metrics from completed trades and equity curve.

        Metrics:
          total_return_pct, annualised_return_pct, sharpe_ratio,
          max_drawdown_pct, win_rate_pct, total_trades,
          profitable_trades, avg_profit, avg_loss
        """
        if not equity_curve:
            return self._zero_metrics()

        eq_values = [e["value"] for e in equity_curve]
        final_value = eq_values[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100

        # Annualised return
        n_days = len(equity_curve)
        ann_return = ((final_value / initial_capital) ** (252 / max(n_days, 1)) - 1) * 100

        # Daily returns for Sharpe
        eq_series = pd.Series(eq_values)
        daily_ret = eq_series.pct_change().dropna()
        sharpe = (
            float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
            if daily_ret.std() > 0
            else 0.0
        )

        # Max drawdown
        roll_max = eq_series.cummax()
        drawdown = (eq_series - roll_max) / roll_max * 100
        max_drawdown = float(drawdown.min())

        # Trade statistics
        total_trades = len(trades)
        if total_trades == 0:
            win_rate = 0.0
            avg_profit = 0.0
            avg_loss = 0.0
        else:
            profits = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
            losses = [t["pnl"] for t in trades if t.get("pnl", 0) <= 0]
            win_rate = len(profits) / total_trades * 100
            avg_profit = sum(profits) / len(profits) if profits else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0

        return {
            "total_return_pct": round(total_return, 2),
            "annualised_return_pct": round(ann_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(win_rate, 1),
            "total_trades": total_trades,
            "profitable_trades": len(profits) if total_trades else 0,
            "avg_profit": round(avg_profit, 2),
            "avg_loss": round(avg_loss, 2),
            "final_value": round(final_value, 2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_trades(
        self,
        signals_df: pd.DataFrame,
        initial_capital: float,
        risk_percent: float,
    ):
        """
        Iterate through signals and simulate a long-only portfolio.
        Returns (trades_list, equity_curve_list).
        """
        if signals_df.empty:
            return [], []

        capital = initial_capital
        position_shares = 0.0
        entry_price = 0.0
        entry_date = None
        trades = []
        equity_curve = []

        for date, row in signals_df.iterrows():
            close = row["close"]
            signal = row["composite_signal"]

            # Portfolio value
            portfolio_value = capital + position_shares * close
            equity_curve.append({"date": date, "value": round(portfolio_value, 2)})

            if signal == "BUY" and position_shares == 0 and capital > 0:
                # Risk-based position sizing
                risk_amount = capital * (risk_percent / 100)
                stop_dist = close * 0.02  # 2% stop
                if stop_dist > 0:
                    shares = min(risk_amount / stop_dist, capital / close)
                    position_shares = round(shares, 6)
                    cost = position_shares * close
                    capital -= cost
                    entry_price = close
                    entry_date = date
                    logger.debug("BUY  %s @ %.2f  shares=%.4f", date, close, position_shares)

            elif signal == "SELL" and position_shares > 0:
                proceeds = position_shares * close
                pnl = proceeds - position_shares * entry_price
                capital += proceeds
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(close, 4),
                        "shares": round(position_shares, 6),
                        "pnl": round(pnl, 2),
                        "return_pct": round(pnl / (position_shares * entry_price) * 100, 2),
                    }
                )
                logger.debug(
                    "SELL %s @ %.2f  P&L=%.2f", date, close, pnl
                )
                position_shares = 0.0
                entry_price = 0.0
                entry_date = None

        # Close any open position at end of period
        if position_shares > 0 and not signals_df.empty:
            last_close = signals_df["close"].iloc[-1]
            proceeds = position_shares * last_close
            pnl = proceeds - position_shares * entry_price
            capital += proceeds
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": signals_df.index[-1],
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(last_close, 4),
                    "shares": round(position_shares, 6),
                    "pnl": round(pnl, 2),
                    "return_pct": round(pnl / (position_shares * entry_price) * 100, 2),
                }
            )

        return trades, equity_curve

    def _get_hist_sentiment(self, ticker: str, date) -> float:
        """Return average sentiment score for 30 days before *date* from cache."""
        try:
            end = pd.Timestamp(date)
            start = end - pd.Timedelta(days=30)
            return self._cache.get_window_score(ticker, start, end)
        except Exception:
            return 0.0

    @staticmethod
    def _offset_date(date_str: str, days: int) -> str:
        dt = datetime.strptime(date_str, "%Y-%m-%d") + pd.Timedelta(days=days)
        return dt.strftime("%Y-%m-%d")

    def _empty_result(self, ticker: str) -> Dict:
        return {
            "ticker": ticker,
            "start_date": "",
            "end_date": "",
            "initial_capital": config.INITIAL_CAPITAL,
            "signals": pd.DataFrame(),
            "trades": [],
            "equity_curve": [],
            "metrics": self._zero_metrics(),
        }

    @staticmethod
    def _zero_metrics() -> Dict:
        return {
            "total_return_pct": 0.0,
            "annualised_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "final_value": config.INITIAL_CAPITAL,
        }
